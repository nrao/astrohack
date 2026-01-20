import os

import numpy as np
import xarray as xr

from casacore import tables as ctables
from numba import njit
from numba.core import types
from numba.typed import Dict
from scipy import spatial

import toolviper.utils.logger as logger

from astrohack.utils.conversion import convert_dict_from_numba
from astrohack.utils.graph import compute_graph_to_mds_tree
from astrohack.utils.tools import get_valid_state_ids
from astrohack.io.point_mds import AstrohackPointFile


def process_extract_pointing(input_params):
    """Top level function to extract subset of pointing table data into a dictionary of xarray data arrays.

    Args:
        input_params(dict): extract_pointing parameters

    Returns:
        AstrohackPointFile: point mds file
    """

    ms_name = input_params["ms_name"]
    pnt_name = input_params["point_name"]
    exclude = input_params["exclude"]

    # Get antenna names and ids
    ctb = ctables.table(
        os.path.join(ms_name, "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    antenna_name = ctb.getcol("NAME")
    ctb.close()

    antenna_id = list(range(len(antenna_name)))

    # Exclude antennas according to user direction
    if exclude:
        if not isinstance(exclude, list):
            exclude = list(exclude)
        for i_ant, antenna in enumerate(exclude):
            if antenna in antenna_name:
                antenna_name.remove(antenna)
                antenna_id.remove(i_ant)

    antenna_id = np.array(antenna_id)

    # Get Holography scans with start and end times.
    ctb = ctables.table(
        ms_name,
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    scan_ids = ctb.getcol("SCAN_NUMBER")
    time = ctb.getcol("TIME")
    ddi = ctb.getcol("DATA_DESC_ID")
    state_ids = ctb.getcol("STATE_ID")
    ctb.close()

    # Get state ids where holography is done
    ctb = ctables.table(
        os.path.join(ms_name, "STATE"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    # scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE sub-table.
    obs_modes = ctb.getcol("OBS_MODE")
    ctb.close()
    mapping_state_ids = get_valid_state_ids(obs_modes)

    mapping_state_ids = np.array(mapping_state_ids)

    # For each ddi get holography scan start and end times:
    scan_time_dict = _extract_scan_time_dict(
        time, scan_ids, state_ids, ddi, mapping_state_ids
    )

    # Create mds file here
    point_mds = AstrohackPointFile.create_from_input_parameters(pnt_name, input_params)
    point_mds.root.attrs["mapping_state_ids"] = mapping_state_ids

    ###########################################################################################
    pnt_params = {
        "ms_name": ms_name,
        "pnt_name": pnt_name,
        "scan_time_dict": scan_time_dict,
        "ant": "all",
    }

    looping_dict = {}
    for i_ant, ant_name in enumerate(antenna_name):
        looping_dict[f"ant_{ant_name}"] = {"id": antenna_id[i_ant], "name": ant_name}

    executed_graph = compute_graph_to_mds_tree(
        looping_dict,
        _make_ant_pnt_chunk,
        pnt_params,
        ["ant"],
        point_mds,
    )
    if executed_graph:
        point_mds.write()
        return point_mds
    else:
        logger.warning("No data to process")
        return None


def _make_ant_pnt_chunk(pnt_params):
    """Extract subset of pointing table data into a dictionary of xarray data arrays. This is written to disk as a
    zarr file. This function processes a chunk the overall data and is managed by Dask.

    Args:
        pnt_params(dict): extract_pointing parameters
    """
    data_dict = pnt_params["data_dict"]
    ms_name = pnt_params["ms_name"]
    scan_time_dict = pnt_params["scan_time_dict"]

    ant_id = data_dict["id"]
    ant_name = data_dict["name"]
    ant_key = pnt_params["this_ant"]

    table_obj = ctables.table(
        os.path.join(ms_name, "POINTING"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    tb = ctables.taql(
        "select DIRECTION, TIME, TARGET, ENCODER, ANTENNA_ID, POINTING_OFFSET from $table_obj WHERE ANTENNA_ID == %s"
        % ant_id
    )

    # NB: Add check if directions reference frame is Azimuth Elevation (AZELGEO)
    try:
        direction = tb.getcol("DIRECTION")[:, 0, :]
        target = tb.getcol("TARGET")[:, 0, :]
        encoder = tb.getcol("ENCODER")
        direction_time = tb.getcol("TIME")
        pointing_offset = tb.getcol("POINTING_OFFSET")[:, 0, :]

    except RuntimeError:
        tb.close()
        logger.warning("Skipping antenna " + str(ant_id) + " no pointing info")

        return 0

    tb.close()
    table_obj.close()

    evaluate_time_samping(direction_time, ant_name)

    pnt_xds = xr.Dataset()
    coords = {"time": direction_time}
    pnt_xds = pnt_xds.assign_coords(coords)

    # Measurement set v2 definition: https://drive.google.com/file/d/1IapBTsFYnUT1qPu_UK09DIFGM81EIZQr/view?usp=sharing
    # DIRECTION: Antenna pointing direction
    pnt_xds["DIRECTION"] = xr.DataArray(direction, dims=("time", "az_el"))

    # ENCODER: The current encoder values on the primary axes of the mount type for the antenna, expressed as a
    # Direction Measure.
    pnt_xds["ENCODER"] = xr.DataArray(encoder, dims=("time", "az_el"))

    # TARGET: This is the true expected position of the source, including all coordinate corrections such as precession,
    # nutation etc.
    pnt_xds["TARGET"] = xr.DataArray(target, dims=("time", "az_el"))

    # POINTING_OFFSET: The a priori pointing corrections applied by the telescope in pointing to the DIRECTION position,
    # optionally expressed as polynomial coefficients.
    pnt_xds["POINTING_OFFSET"] = xr.DataArray(pointing_offset, dims=("time", "az_el"))

    # Calculate directional cosines (l,m) which are used as the gridding locations.
    # See equations 8,9 in https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf.
    # TARGET: A_s, E_s (target source position)
    # DIRECTION: A_a, E_a (Antenna's pointing direction)

    # ## NB: Is VLA's definition of Azimuth the same for ALMA, MeerKAT, etc.? (positive for a clockwise rotation from
    # north, viewed from above) ## NB: Compare with calculation using WCS in astropy.
    l_points = np.cos(target[:, 1]) * np.sin(target[:, 0] - direction[:, 0])
    m_points = np.sin(target[:, 1]) * np.cos(direction[:, 1]) - np.cos(
        target[:, 1]
    ) * np.sin(direction[:, 1]) * np.cos(target[:, 0] - direction[:, 0])

    pnt_xds["DIRECTIONAL_COSINES"] = xr.DataArray(
        np.array([l_points, m_points]).T, dims=("time", "lm")
    )

    """
    Notes from ASDM (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    Science Data Model Binary Data Format:    https://drive.google.com/file/d/1PMrZFbkrMVfe57K6AAh1dR1FalS35jP2/view
        
    A - ASDM, MS - MS
    
    A_encoder = The values measured from the antenna. They may be however affected by metrology, if applied. Note 
    that for ALMA this column will contain positions obtained using the AZ POSN RSP and EL POSN RSP monitor points of 
    the ACU and not the GET AZ ENC and GET EL ENC monitor points (as these do not include the metrology corrections). 
    It is agreed that the the vendor pointing model will never be applied. AZELNOWAntenna.position 
    
    A_pointing_direction : This is the commanded direction of the antenna. It is obtained by adding the target and 
    offset columns, and then applying the pointing model referenced by PointingModelId. The pointing model can be the 
    composition of the absolute pointing model and of a local pointing model. In that case their coefficients will 
    both be in the PointingModel table. A_target : This is the field center direction (as given in the Field Table), 
    possibly affected by the optional antenna-based sourceOffset. This column is in horizontal coordinates. 
    
    AZELNOWAntenna.position A_offset : Additional offsets in horizontal coordinates (usually meant for measuring the 
    pointing corrections, mapping the antenna beam, ...). AZELNOWAntenna.positiontarget A_sourceOffset : Optionally, 
    the antenna-based mapping offsets in the field. These are in the equatorial system, and used, for instance, 
    in on-the-fly mapping when the antennas are driven independently across the field.
                    
                    
    M_direction = rotate(A_target,A_offset) #A_target is rotated to by A_offset
    if withPointingCorrection:
        M_direction = rotate(A_target,A_offset) + (A_encoder - A_pointing_direction)
        
    M_target = A_target
    M_pointing_offset = A_offset
    M_encoder = A_encoder
    
    From the above description I suspect encoder should be used instead of direction, however for the VLA mapping 
    antenna data no grid pattern appears (ALMA data does not have this problem)."""

    # Detect during which scans an antenna is mapping by averaging the POINTING_OFFSET radius.
    time_tree = spatial.KDTree(direction_time[:, None])  # Use for nearest interpolation

    mapping_scans_obs_dict = {}

    for ddi_id, ddi in scan_time_dict.items():
        map_scans_dict = {}
        map_id = 0

        for scan_id, scan_time in ddi.items():
            _, time_index = time_tree.query(scan_time[:, None])

            pointing_offset_scan_slice = pnt_xds["POINTING_OFFSET"].isel(
                time=slice(time_index[0], time_index[1])
            )

            avg_lm_dist = (
                np.sqrt(
                    pointing_offset_scan_slice.isel(az_el=0) ** 2
                    + pointing_offset_scan_slice.isel(az_el=1) ** 2
                )
            ).mean()

            if avg_lm_dist > 10**-12:  # Antenna is mapping since lm is non-zero
                if ("map_" + str(map_id)) in map_scans_dict:
                    map_scans_dict["map_" + str(map_id)].append(scan_id)

                else:
                    map_scans_dict["map_" + str(map_id)] = [scan_id]

            else:
                map_id = map_id + 1

        mapping_scans_obs_dict["ddi_" + str(ddi_id)] = map_scans_dict

    pnt_xds.attrs["mapping_scans_obs_dict"] = [mapping_scans_obs_dict]
    ###############

    pnt_xds.attrs["ant_name"] = ant_name

    return xr.DataTree(dataset=pnt_xds, name=ant_key)


def _extract_scan_time_dict(time, scan_ids, state_ids, ddi_ids, mapping_state_ids):
    """
    [ddi][scan][start, stop]
    """
    scan_time_dict = _extract_scan_time_dict_jit(
        time, scan_ids, state_ids, ddi_ids, mapping_state_ids
    )

    # This section cleans up the case of when there was an issue with the scan time data. If the scan start and end
    # times are the same the mapping(reference) state identification does not work. For this reason, scans containing
    # instance of this are removed and any empty ddi(s) are dropped as well.
    drops = {}

    for ddi in scan_time_dict.keys():
        drops[ddi] = []

        for scan in scan_time_dict[ddi].keys():
            if scan_time_dict[ddi][scan][0] == scan_time_dict[ddi][scan][1]:
                drops[ddi].append(scan)

    for ddi, scans in drops.items():
        for scan in scans:
            del scan_time_dict[ddi][scan]

        if len(scan_time_dict[ddi]) == 0:
            del scan_time_dict[ddi]

    return scan_time_dict


@convert_dict_from_numba
@njit(cache=False, nogil=True)
def _extract_scan_time_dict_jit(time, scan_ids, state_ids, ddi_ids, mapping_state_ids):
    """For each ddi get holography scan start and end times. A holography scan is detected when a scan_ids appears in
    mapping_state_ids.

    """
    dict_template = Dict.empty(
        key_type=types.int64,
        value_type=np.zeros(2, dtype=types.float64),
    )

    scan_time_dict = Dict.empty(
        key_type=types.int64,
        value_type=dict_template,
    )

    mapping_scans = set()

    for i_scan, scan_num in enumerate(scan_ids):
        scan_num = types.int64(scan_num)
        time_val = time[i_scan]
        ddi = ddi_ids[i_scan]

        state_id = state_ids[i_scan]

        if state_id in mapping_state_ids:
            mapping_scans.add(scan_num)
            if ddi in scan_time_dict:
                if scan_num in scan_time_dict[ddi]:
                    if scan_time_dict[ddi][scan_num][0] > time_val:
                        scan_time_dict[ddi][scan_num][0] = time_val

                    if scan_time_dict[ddi][scan_num][1] < time_val:
                        scan_time_dict[ddi][scan_num][1] = time_val

                else:
                    scan_time_dict[ddi][scan_num] = np.array([time_val, time_val])

            else:
                scan_time_dict[ddi] = {scan_num: np.array([time_val, time_val])}

    return scan_time_dict


def evaluate_time_samping(
    time_sampling, data_label, threshold=0.01, expected_interval=0.1
):
    bin_sz = expected_interval / 4
    time_bin_edge = np.arange(-bin_sz / 2, 2.5 * expected_interval, bin_sz)
    time_bin_axis = time_bin_edge[:-1] + bin_sz / 2
    i_mid = int(np.argmin(np.abs(time_bin_axis - expected_interval)))

    intervals = np.diff(time_sampling)
    hist, edges = np.histogram(intervals, bins=time_bin_edge)
    n_total = np.sum(hist)
    outlier_fraction = 1 - hist[i_mid] / n_total

    if outlier_fraction > threshold:
        logger.warning(
            f"{data_label} pointing table has {100*outlier_fraction:.2}% of data with irregular "
            f"time sampling"
        )
