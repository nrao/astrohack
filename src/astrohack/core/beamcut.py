import numpy
import toolviper.utils.logger as logger
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import astropy
import xarray as xr

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.utils.file import load_holog_file
from astrohack.utils import (
    create_dataset_label,
    convert_unit,
    sig_2_fwhm,
    format_frequency,
    format_value_unit,
    to_db,
    create_pretty_table,
)
from astrohack.visualization import create_figure_and_axes, scatter_plot, close_figure
from astrohack.visualization.plot_tools import set_y_axis_lims_from_default

lnbr = "\n"
spc = " "
quack_chans = 4


###########################################################
### Working Chunks
###########################################################
def process_beamcut_chunk(beamcut_chunk_params):
    """
    Ingests a holog_xds containing beamcuts and produces a beamcut_xdtree containing the cuts separated in xdses.

    :param beamcut_chunk_params: Parameter dictionary with inputs
    :type beamcut_chunk_params: dict

    :return: Beamcut_xdtree containing the different cuts for this antenna and DDI.
    :rtype: xr.DataTree
    """
    ddi = beamcut_chunk_params["this_ddi"]
    antenna = beamcut_chunk_params["this_ant"]

    _, ant_data_dict = load_holog_file(
        beamcut_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=antenna,
        ddi_id=ddi,
    )
    # This assumes that there will be no more than one mapping
    input_xds = ant_data_dict[ddi]["map_0"]
    datalabel = create_dataset_label(antenna, ddi)
    logger.info(f"processing {datalabel}")

    cut_xdtree = _extract_cuts_from_visibilities(input_xds, antenna, ddi)

    _beamcut_multi_lobes_gaussian_fit(cut_xdtree, datalabel)

    destination = beamcut_chunk_params["destination"]
    if destination is not None:
        logger.info(f"Producing plots for {datalabel}")
        plot_beamcut_in_amplitude_chunk(beamcut_chunk_params, cut_xdtree)
        plot_beamcut_in_attenuation_chunk(beamcut_chunk_params, cut_xdtree)
        plot_cuts_in_lm_chunk(beamcut_chunk_params, cut_xdtree)
        create_report_chunk(beamcut_chunk_params, cut_xdtree)
        logger.info(f"Completed plots for {datalabel}")

    return cut_xdtree


def plot_beamcut_in_amplitude_chunk(par_dict, cut_xdtree=None):
    """
    Produce Amplitude beam cut plots from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :return: None
    :rtype: NoneType
    """
    if cut_xdtree is None:
        cut_xdtree = par_dict["xdt_data"]
    n_cuts = len(cut_xdtree.children.values())
    # Loop over cuts
    fig, axes = create_figure_and_axes([12, 1 + n_cuts * 4], [n_cuts, 2])
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        _plot_single_cut_in_amplitude(cut_xds, axes[icut, :], par_dict)

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("amplitude", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


def plot_beamcut_in_attenuation_chunk(par_dict, cut_xdtree=None):
    """
    Produce attenuation beam cut plots from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :return: None
    :rtype: NoneType
    """
    if cut_xdtree is None:
        cut_xdtree = par_dict["xdt_data"]
    n_cuts = len(cut_xdtree.children.values())
    # Loop over cuts
    fig, axes = create_figure_and_axes([6, 1 + n_cuts * 4], [n_cuts, 1])
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        _plot_single_cut_in_attenuation(cut_xds, axes[icut], par_dict)

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("attenuation", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


def plot_cuts_in_lm_chunk(par_dict, cut_xdtree=None):
    """
    Produce plot of LM offsets in all cuts for antenna and ddi xd tree.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :return: None
    :rtype: NoneType
    """
    if cut_xdtree is None:
        cut_xdtree = par_dict["xdt_data"]
    _plot_cuts_in_lm_sub(cut_xdtree, par_dict)


def create_report_chunk(
    par_dict, cut_xdtree=None, spacing=2, item_marker="-", precision=3
):
    """
    Produce a report on beamcut fit results from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling report aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :param spacing: Identation
    :type spacing: int

    :param item_marker: Character to denote a different item in a list
    :type item_marker: str

    :param precision: Number of decimal places to include in table results
    :type precision: int

    :return: None
    :rtype: NoneType
    """
    if cut_xdtree is None:
        cut_xdtree = par_dict["xdt_data"]
    outstr = f"{item_marker}{spc}"
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")
    summary = cut_xdtree.attrs["summary"]

    items = [
        "Id",
        f"Center [{lm_unit}]",
        "Amplitude [ ]",
        f"FWHM [{lm_unit}]",
        "Attenuation [dB]",
    ]
    outstr += _create_beamcut_header(summary, par_dict) + 2 * lnbr
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        sub_title = _make_parallel_hand_sub_title(cut_xds.attrs)
        for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
            outstr += f"{spacing*spc}{item_marker}{spc}{parallel_hand} {sub_title}, Beam fit results:{lnbr}"
            table = create_pretty_table(items, "c")
            fit_pars = cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"]
            centers = fit_pars[0::3]
            amps = fit_pars[1::3]
            fwhms = fit_pars[2::3]
            max_amp = np.max(cut_xds[f"{parallel_hand}_amplitude"].values)

            for i_peak in range(cut_xds.attrs[f"{parallel_hand}_n_peaks"]):

                table.add_row(
                    [
                        f"{i_peak+1})",  # Id
                        f"{lm_fac*centers[i_peak]:.{precision}f}",  # center
                        f"{amps[i_peak]:.{precision}f}",  # Amp
                        f"{lm_fac*fwhms[i_peak]:.{precision}f}",  # FWHM
                        f"{to_db(amps[i_peak]/max_amp):.{precision}f}",  # Attenuation
                    ]
                )
            for line in table.get_string().splitlines():
                outstr += 2 * spacing * spc + line + lnbr
            outstr += lnbr

    with open(_file_name_factory("report", par_dict), "w") as outfile:
        outfile.write(outstr)


###########################################################
### Data IO
###########################################################
def _file_name_factory(file_type, par_dict):
    """
    Generate filenames from file type and execution parameters

    :param file_type: File type description
    :type file_type: str

    :param par_dict: Paremeter dictionary containing destination, antenna and ddi parameters
    :type par_dict: dict

    :return: Filename
    :rtype: str
    """
    destination = par_dict["destination"]
    antenna = par_dict["this_ant"]
    ddi = par_dict["this_ddi"]
    if file_type in ["attenuation", "amplitude", "lm_offsets"]:
        ext = "png"
    elif file_type == "report":
        ext = "txt"
    else:
        raise ValueError("Invalid file type")
    return f"{destination}/beamcut_{file_type}_{antenna}_{ddi}.{ext}"


###########################################################
### Data extraction
###########################################################
def _time_scan_selection(scan_time_ranges, time_axis):
    """
    Produce scan based time selection
    :param scan_time_ranges: MS derived scan time ranges
    :type scan_time_ranges: list

    :param time_axis: Visibilities time axis
    :type time_axis: numpy.array

    :return: Selection in time for each scan.
    :rtype: numpy.array(dtype=bool)
    """
    time_selections = []
    for scan_time_range in scan_time_ranges:
        time_selection = np.logical_and(
            time_axis >= scan_time_range[0], time_axis < scan_time_range[1]
        )
        time_selections.append(time_selection)
    return time_selections


def _extract_cuts_from_visibilities(input_xds, antenna, ddi):
    """
    Creates data tree containing the different cuts from a holog xds.

    :param input_xds: holog xds containing visibilities with beam cuts.
    :type input_xds: xarray.Dataset

    :param antenna: Antenna key
    :type antenna: str

    :param ddi: DDI key
    :type ddi: str

    :return: Data tree containing the beamcut xdses.
    :rtype: xarray.DataTree
    """
    cut_xdtree = xr.DataTree(name=f"{antenna}-{ddi}")
    scan_time_ranges = input_xds.attrs["scan_time_ranges"]
    scan_list = input_xds.attrs["scan_list"]
    cut_xdtree.attrs["summary"] = input_xds.attrs["summary"]

    lm_offsets = input_xds.DIRECTIONAL_COSINES.values
    time_axis = input_xds.time.values
    corr_axis = input_xds.pol.values
    visibilities = input_xds.VIS.values
    weights = input_xds.WEIGHT.values

    nchan = visibilities.shape[1]
    fchan = 4
    lchan = int(nchan - fchan)
    for iscan, scan_number in enumerate(scan_list):
        scan_time_range = scan_time_ranges[iscan]
        time_selection = np.logical_and(
            time_axis >= scan_time_range[0], time_axis < scan_time_range[1]
        )
        time = time_axis[time_selection]
        this_lm_offsets = lm_offsets[time_selection, :]

        lm_angle, lm_dist, direction, xlabel = (
            _cut_direction_determination_and_label_creation(this_lm_offsets)
        )
        hands_dict = _get_parallel_hand_indexes(corr_axis)

        avg_vis = np.average(
            visibilities[time_selection, fchan:lchan, :],
            axis=1,
            weights=weights[time_selection, fchan:lchan, :],
        )
        avg_wei = np.average(weights[time_selection, fchan:lchan, :], axis=1)

        avg_time = np.average(time) * convert_unit("sec", "day", "time")
        timestr = astropy.time.Time(avg_time, format="mjd").to_value(
            "iso", subfmt="date_hm"
        )

        xds = xr.Dataset()
        coords = {"lm_dist": lm_dist, "time": time}

        xds.attrs.update(
            {
                "scan_number": scan_number,
                "lm_angle": lm_angle,
                "available_corrs": list(hands_dict.keys()),
                "direction": direction,
                "xlabel": xlabel,
                "time_string": timestr,
            }
        )

        xds["lm_offsets"] = xr.DataArray(this_lm_offsets, dims=["time", "lm"])
        all_corr_ymax = 1e-34
        for parallel_hand, icorr in hands_dict.items():
            amp = np.abs(avg_vis[:, icorr])
            maxamp = np.max(amp)
            if maxamp > all_corr_ymax:
                all_corr_ymax = maxamp
            xds[f"{parallel_hand}_amplitude"] = xr.DataArray(amp, dims="lm_dist")
            xds[f"{parallel_hand}_phase"] = xr.DataArray(
                np.angle(avg_vis[:, icorr]), dims="lm_dist"
            )
            xds[f"{parallel_hand}_weight"] = xr.DataArray(
                avg_wei[:, icorr], dims="lm_dist"
            )
        xds.attrs.update({"all_corr_ymax": all_corr_ymax})
        cut_xdtree = cut_xdtree.assign(
            {
                f"cut_{iscan}": xr.DataTree(
                    dataset=xds.assign_coords(coords), name=f"cut_{iscan}"
                )
            }
        )

    return cut_xdtree


def _cut_direction_determination_and_label_creation(lm_offsets, angle_unit="deg"):
    """
    Determines cut's direction using a linear regression between L and M offsets.

    :param lm_offsets: Array containing the cut's L and M offsets over type expected to be of shape [n_time, lm]
    :type lm_offsets: numpy.ndarray

    :param angle_unit: Unit to represent cut's direction in a mixed cut.
    :type angle_unit: str

    :return: Tuple containing the cuts direction angle in the sky, distance from center for each point, direction \
    label, and x-axis label for plots.
    :rtype: tuple([float, numpy.array, str, str])

    """
    dx = lm_offsets[-1, 0] - lm_offsets[0, 0]
    dy = lm_offsets[-1, 1] - lm_offsets[0, 1]

    # determine where to flip signal of distances
    lm_dist = np.sqrt(lm_offsets[:, 0] ** 2 + lm_offsets[:, 1] ** 2)
    imin_lm = np.argmin(lm_dist)
    x_min, y_min = lm_offsets[imin_lm, :]
    lm_dist[:imin_lm] = -lm_dist[:imin_lm]

    if np.isclose(dx, dy, rtol=3e-1):  # X case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi / 2
        direction = "mixed cut("
        if dy < 0 and dx < 0:
            direction += "NW -> SE"
            # Fix the sign of the minimum
            if x_min > 0:
                lm_dist[imin_lm] *= -1

        elif dy < 0 < dx:
            # Fix the sign of the minimum
            if x_min < 0:
                lm_dist[imin_lm] *= -1
            direction += "NE -> SW"

        elif dy > 0 > dx:
            # Fix the sign of the minimum
            if x_min > 0:
                lm_dist[imin_lm] *= -1
            direction += "SW -> NE"

        else:
            # Fix the sign of the minimum
            if x_min < 0:
                lm_dist[imin_lm] *= -1
            direction += "SE -> NW"

        direction += (
            r", $\theta$ = "
            + f"{format_value_unit(convert_unit('rad', angle_unit, 'trigonometric')*lm_angle, angle_unit)}"
        )
        xlabel = "Mixed offset"
    elif np.abs(dy) > np.abs(dx):  # Elevation case
        result = linregress(lm_offsets[:, 1], lm_offsets[:, 0])
        lm_angle = np.arctan(result.slope)
        # Fix the sign of the minimum
        if y_min < 0:
            lm_dist[imin_lm] *= -1
        direction = "El. cut ("
        if dy < 0:
            direction += "N -> S"
            lm_dist *= -1  # Flip as sense is negative
        else:
            direction += "S -> N"
        xlabel = "Elevation offset"
    else:  # Azimuth case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi / 2
        # Fix the sign of the minimum
        if x_min < 0:
            lm_dist[imin_lm] *= -1
        direction = "Az. cut ("
        if dx > 0:
            direction += "E -> W"
        else:
            direction += "W -> E"
            lm_dist *= -1  # Flip as sense is negative
        xlabel = "Azimuth offset"

    direction += ")"

    return lm_angle, lm_dist, direction, xlabel


def _get_parallel_hand_indexes(corr_axis):
    """
    Get the indices of parallel hands along the correlation axis.

    :param corr_axis: Visibilities correlation axis
    :rtype: numpy.array(str)

    :return: Dictionary containing the parallel hands and their indices
    :rtype: dict
    """
    if "L" in corr_axis[0] or "R" in corr_axis[0]:
        parallel_hands = ["RR", "LL"]
    else:
        parallel_hands = ["XX", "YY"]

    hands_dict = {}
    for icorr, corr in enumerate(corr_axis):
        if corr in parallel_hands:
            hands_dict[corr] = icorr
    return hands_dict


###########################################################
### Multiple side lobe Gaussian fitting
###########################################################
def _fwhm_gaussian(x_axis, x_off, amp, fwhm):
    """
    Returns a gaussian of the same shape as x_axis with fwhm instead of sigma as the input parameter

    :param x_axis: X-axis of the beam cut data
    :type x_axis: numpy.array

    :param x_off: X offset of the center of the gaussian
    :type x_off: float

    :param amp: Amplitude of the gaussian
    :type amp: float

    :param fwhm: Full width at half maximum of the gaussian
    :type fwhm: float

    :return: Gaussian evaluated at x_axis points
    :rtype: numpy.array
    """
    sigma = fwhm / sig_2_fwhm
    return amp * np.exp(-((x_axis - x_off) ** 2) / (2 * sigma**2))


def _build_multi_gaussian_initial_guesses(
    x_data, y_data, pb_fwhm, min_dist_fraction=1.3
):
    """
    Build initial guesses array for a multi gaussian fitting from X and Y axes heuristics

    :param x_data: Fit's X-axis data.
    :type x_data: numpy.array

    :param y_data: Fit's Y-axis data.
    :type  y_data: numpy.array

    :param pb_fwhm: Estimated FWHM of the primary beam
    :type pb_fwhm: float

    :param min_dist_fraction: Fraction of pb_fwhm to use as estimate for the minimal peak distance
    :type min_dist_fraction: float

    :return: Tuple containing the initial_guesses, bounds and number of peaks to fit
    :rtype: tuple([list, list([list]), int])
    """
    initial_guesses = []
    lower_bounds = []
    upper_bounds = []
    step = float(np.median(np.diff(x_data)))
    min_dist = np.abs(min_dist_fraction * pb_fwhm / step)
    peaks, _ = find_peaks(y_data, distance=min_dist)
    dx = x_data[-1] - x_data[0]
    if dx < 0:
        peaks = peaks[::-1]
    for ipeak in peaks:
        initial_guesses.extend([x_data[ipeak], y_data[ipeak], pb_fwhm])
        lower_bounds.extend([-np.inf, 0, 0])
        upper_bounds.extend([np.inf, np.inf, np.inf])
    bounds = (lower_bounds, upper_bounds)
    return initial_guesses, bounds, len(peaks)


def _multi_gaussian(xdata, *args):
    """
    Produces a multiple gaussian Y array from parameters derived fromm list of arguments

    :param xdata: X-axis data
    :type xdata: numpy.array

    :param args: List of gaussian parameters [x_off_0, amp_0, fwhm_0, ..., x_off_n, amp_n, fwhm_n]
    :type args: list([float])

    :return: Multiple gaussian evaluated at x-axis points
    :rtype: numpy.array
    """
    nargs = len(args)
    if nargs % 3 != 0:
        raise ValueError("Number of arguments should be multiple of 3")
    y_values = np.zeros_like(xdata)
    for iarg in range(0, nargs, 3):
        y_values += _fwhm_gaussian(xdata, args[iarg], args[iarg + 1], args[iarg + 2])
    return y_values


def _perform_curvefit_with_given_functions(
    x_data, y_data, initial_guesses, bounds, fit_func, datalabel, maxit=50000
):
    """
    Invoke scipy optimize curve_fit with customized parameters

    :param x_data: x-axis data for the curve fit
    :type x_data: numpy.array

    :param y_data: y-axis data to be fitted
    :type y_data: numpy.array

    :param initial_guesses: list of initial guesses
    :type initial_guesses: list

    :param bounds: List containing the lists of lower and upper bounds for each parameter
    :type bounds: list([list])

    :param fit_func: Function with which to fit the parameters.
    :type fit_func: function

    :param datalabel: Data label for messaging
    :type datalabel: str

    :param maxit: Maximum number of iterations
    :type maxit: int

    :return: Tuple containing sucess flag and fit results (NaNs if fit failed)
    :rtype: tuple([bool, numpy.array])
    """
    try:
        results = curve_fit(
            fit_func,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=bounds,
            maxfev=int(maxit),
        )
        fit_pars = results[0]
        return True, fit_pars
    except RuntimeError:
        logger.warning(f"{fit_func.__name__} fit to lobes failed for {datalabel}.")
        return False, np.full_like(initial_guesses, np.nan)


def _identify_pb_and_sidelobes_in_fit(datalabel, x_data, fit_pars):
    """
    Identify primary beam and first sidelobes in fit using expected beam shape heuristics.

    :param datalabel: Data label for messaging
    :type datalabel: str

    :param x_data: X-axis data
    :type x_data: numpy.array

    :param fit_pars: Fit parameters
    :type fit_pars: numpy.array

    :return: Tuple containing: Number of peaks in beam, filtered fit parameters, primary beam center offset, primary \
    beam measured fwhm, ratio between left and right first sidelobes
    :rtype: tuple([int, numpy.array, float, float, float])
    """
    centers = fit_pars[0::3]
    amps = fit_pars[1::3]
    fwhms = fit_pars[2::3]

    # select fits that are within x_data
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    selection = ~np.logical_or(centers < x_min, centers > x_max)

    # apply selection
    centers = centers[selection]
    amps = amps[selection]
    fwhms = fwhms[selection]

    # Reconstruct fit metadata
    n_peaks = centers.shape[0]
    fit_pars = np.zeros((3 * n_peaks))
    fit_pars[0::3] = centers
    fit_pars[1::3] = amps
    fit_pars[2::3] = fwhms

    # This assumes the primary beam is the closest to the center, which is expected
    i_pb_cen = np.argmin(np.abs(centers))
    # This assumes the primary beam is the strongest
    i_pb_amp = np.argmax(amps)
    pb_problem = i_pb_cen != i_pb_amp

    if pb_problem:
        logger.warning(f"Cannot reliably identify primary beam for {datalabel}.")
        pb_center, pb_fwhm, first_side_lobe_ratio = np.nan, np.nan, np.nan

    else:
        pb_fwhm = fwhms[i_pb_cen]
        pb_center = centers[i_pb_cen]

        pb_cen = centers[i_pb_cen]
        i_closest_to_center = np.argsort(np.abs(centers - pb_cen))
        if centers[i_closest_to_center[1]] < 0:
            i_lsl = i_closest_to_center[1]
            i_rsl = i_closest_to_center[2]
        else:
            i_lsl = i_closest_to_center[2]
            i_rsl = i_closest_to_center[1]
        left_first_sl_amp = amps[i_lsl]
        right_first_sl_amp = amps[i_rsl]
        first_side_lobe_ratio = left_first_sl_amp / right_first_sl_amp

    return n_peaks, fit_pars, pb_center, pb_fwhm, first_side_lobe_ratio


def _beamcut_multi_lobes_gaussian_fit(cut_xdtree, datalabel):
    """
    Execute multi gaussian fit to beam cut data.

    :param cut_xdtree: Datatree containing a single beam cut.
    :type cut_xdtree: xarray.DataTree

    :param datalabel: Data label for messaging
    :type datalabel: str

    :return: None
    :rtype: NoneType
    """
    # Get the summary from the first cut, but it should be equal anyway
    summary = cut_xdtree.attrs["summary"]
    wavelength = summary["spectral"]["rep. wavelength"]
    telescope = get_proper_telescope(
        summary["general"]["telescope name"], summary["general"]["antenna name"]
    )
    primary_fwhm = 1.2 * wavelength / telescope.diameter

    for cut_xds in cut_xdtree.children.values():
        x_data = cut_xds["lm_dist"].values
        for parallel_hand in cut_xds.attrs["available_corrs"]:
            y_data = cut_xds[f"{parallel_hand}_amplitude"]
            this_corr_data_label = (
                f'{datalabel}, {cut_xds.attrs["direction"]}, corr = {parallel_hand}'
            )
            initial_guesses, bounds, n_peaks = _build_multi_gaussian_initial_guesses(
                x_data, y_data, primary_fwhm
            )
            fit_succeeded, fit_pars = _perform_curvefit_with_given_functions(
                x_data,
                y_data,
                initial_guesses,
                bounds,
                _multi_gaussian,
                this_corr_data_label,
            )

            if fit_succeeded:
                fit = _multi_gaussian(x_data, *fit_pars)
                n_peaks, fit_pars, pb_center, pb_fwhm, first_side_lobe_ratio = (
                    _identify_pb_and_sidelobes_in_fit(
                        this_corr_data_label, x_data, fit_pars
                    )
                )
            else:
                pb_center, pb_fwhm, first_side_lobe_ratio = np.nan, np.nan, np.nan
                fit = np.full_like(y_data, np.nan)

            cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"] = fit_pars
            cut_xds.attrs[f"{parallel_hand}_n_peaks"] = n_peaks
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] = pb_fwhm
            cut_xds.attrs[f"{parallel_hand}_pb_center"] = pb_center
            cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"] = (
                first_side_lobe_ratio
            )
            cut_xds.attrs[f"{parallel_hand}_fit_succeeded"] = fit_succeeded

            cut_xds[f"{parallel_hand}_amp_fit"] = xr.DataArray(fit, dims="lm_dist")
    return


###########################################################
### Plot utilities
###########################################################
def _add_secondary_beam_hpbw_x_axis_to_plot(pb_fwhm, ax):
    """
    Add a secondary X axis on top of the figure representing the LM distances in primary beam FWHMs.

    :param pb_fwhm: Primary beam FWHM
    :type pb_fwhm: float

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :return: None
    :rtype: NoneType
    """
    if np.isnan(pb_fwhm):
        return
    sec_x_axis = ax.secondary_xaxis(
        "top", functions=(lambda x: x * 1.0, lambda xb: 1 * xb)
    )
    sec_x_axis.set_xlabel("Offset in Primary Beam HPBWs\n")
    sec_x_axis.set_xticks([])
    y_min, y_max = ax.get_ylim()
    x_lims = np.array(ax.get_xlim())
    pb_min, pb_max = np.ceil(x_lims / pb_fwhm)
    beam_offsets = np.arange(pb_min, pb_max, 1, dtype=int)

    for itk in beam_offsets:
        ax.axvline(itk * pb_fwhm, color="k", linestyle="--", linewidth=0.5)
        ax.text(itk * pb_fwhm, y_max, f"{itk:d}", va="bottom", ha="center")


def _add_lobe_identification_to_plot(ax, centers, peaks, y_off):
    """
    Add gaussians identification to plot

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :param centers: Gaussian centers
    :type centers: list, numpy.array

    :param peaks: Gaussian peaks
    :type peaks: list, numpy.array

    :param y_off: Y offset to add peak Ids
    :type y_off: float

    :return: None
    :rtype: NoneType
    """
    for i_peak, peak in enumerate(peaks):
        ax.text(centers[i_peak], peak + y_off, f"{i_peak+1})", ha="center", va="bottom")


def _add_beam_parameters_box(
    ax,
    pb_center,
    pb_fwhm,
    sidelobe_ratio,
    lm_unit,
    alpha=0.8,
    x_pos=0.05,
    y_pos=0.95,
    attenuation_plot=False,
):
    """
    Add text bos with beam parameters

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :param pb_center: Primary beam center offset
    :type pb_center: float

    :param pb_fwhm: Primary beam FWHM
    :type pb_fwhm: float

    :param sidelobe_ratio: First side lobe ratio
    :type sidelobe_ratio: float

    :param lm_unit: L/M axis unit
    :type lm_unit: str

    :param alpha: Opacity of text box
    :type alpha: float

    :param x_pos: Relative x position of the text box
    :type x_pos: float

    :param y_pos: Relative y position of the text box
    :type y_pos: float

    :param attenuation_plot: Is this an attenuation plot?
    :type attenuation_plot: bool

    :return: None
    :rtype: NoneType
    """
    if attenuation_plot:
        head = "avg "
    else:
        head = ""
    pars_str = f"{head}PB off. = {format_value_unit(pb_center, lm_unit, 3)}\n"
    pars_str += f"{head}PB FWHM = {format_value_unit(pb_fwhm, lm_unit, 3)}\n"
    pars_str += f"{head}FSLR = {format_value_unit(to_db(sidelobe_ratio), 'dB', 2)}"
    bounds_box = dict(boxstyle="square", facecolor="white", alpha=alpha)
    ax.text(
        x_pos,
        y_pos,
        pars_str,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=bounds_box,
    )


###########################################################
### Plot correlation subroutines
###########################################################
def _plot_single_cut_in_amplitude(cut_xds, axes, par_dict):
    """
    Plot a single beam cut in amplitude with each correlation in a different panel

    :param cut_xds: xarray dataset containing the beamcut
    :type cut_xds: xarray.Dataset

    :param axes: numpy array with the Matplotlib Axes objects for the different panels
    :type axes: numpy.array([Matplotlib.axes.Axes])

    :param par_dict: Parameter dictionary containing plot configuration
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    # Init
    sub_title = _make_parallel_hand_sub_title(cut_xds.attrs)
    max_amp = cut_xds.attrs["all_corr_ymax"]
    y_off = 0.05 * max_amp
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")

    # Loop over correlations
    for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
        # Init labels
        this_ax = axes[i_corr]
        x_data = lm_fac * cut_xds["lm_dist"].values
        y_data = cut_xds[f"{parallel_hand}_amplitude"].values
        fit_data = cut_xds[f"{parallel_hand}_amp_fit"].values
        xlabel = f"{cut_xds.attrs['xlabel']} [{lm_unit}]"
        ylabel = f"{parallel_hand} Amplitude [ ]"

        # Call plotting tool
        if cut_xds.attrs[f"{parallel_hand}_fit_succeeded"]:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                model=fit_data,
                model_marker="",
                title=sub_title,
                data_marker="+",
                residuals_marker=".",
                model_linestyle="-",
                data_label=f"{parallel_hand} data",
                model_label=f"{parallel_hand} fit",
                data_color="red",
                model_color="blue",
                residuals_color="black",
                legend_location="upper right",
            )

            # Add fit peak identifiers
            centers = lm_fac * np.array(
                cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][0::3]
            )
            amps = np.array(cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][1::3])

            _add_lobe_identification_to_plot(
                this_ax,
                centers,
                amps,
                y_off,
            )
        else:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                title=sub_title,
                data_marker="+",
                data_label=f"{parallel_hand} data",
                data_color="red",
                legend_location="upper right",
            )

        # equalize Y scale between correlations
        set_y_axis_lims_from_default(
            this_ax, par_dict["y_scale"], (-y_off, max_amp + 3 * y_off)
        )

        _add_secondary_beam_hpbw_x_axis_to_plot(
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac, this_ax
        )

        # Add bounded box with Beam parameters
        _add_beam_parameters_box(
            this_ax,
            cut_xds.attrs[f"{parallel_hand}_pb_center"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"],
            lm_unit,
        )


def _plot_single_cut_in_attenuation(cut_xds, ax, par_dict):
    """
    Plot a single beam cut in attenuation with superposed correlations

    :param cut_xds: xarray dataset containing the beamcut
    :type cut_xds: xarray.Dataset

    :param ax: Matplotlib Axes object
    :type ax: Matplotlib.axes.Axes

    :param par_dict: Parameter dictionary containing plot configuration
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    sub_title = _make_parallel_hand_sub_title(cut_xds.attrs)
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")
    corr_colors = ["blue", "red"]

    min_attenuation = 1e34
    pb_center = 0.0
    pb_fwhm = 0.0
    fsl_ratio = 0.0
    xlabel = f"{cut_xds.attrs['xlabel']} [{lm_unit}]"
    ylabel = f"Attenuation [dB]"

    # Loop over correlations
    n_data = 0
    for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
        # Init labels
        x_data = lm_fac * cut_xds["lm_dist"].values
        amps = cut_xds[f"{parallel_hand}_amplitude"].values
        max_amp = np.max(amps)
        y_data = to_db(amps / max_amp)
        y_min = np.min(y_data)
        if y_min < min_attenuation:
            min_attenuation = y_min

        ax.plot(
            x_data,
            y_data,
            label=parallel_hand,
            color=corr_colors[i_corr],
            marker=".",
            ls="",
        )

        if not np.isnan(pb_center):
            pb_center += cut_xds.attrs[f"{parallel_hand}_pb_center"]
            pb_fwhm += cut_xds.attrs[f"{parallel_hand}_pb_fwhm"]
            fsl_ratio += cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"]
            n_data += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(sub_title)
    ax.legend(loc="upper right")

    pb_center /= n_data
    pb_fwhm /= n_data
    fsl_ratio /= n_data
    # equalize Y scale between correlations
    y_off = 0.1 * np.abs(min_attenuation)
    set_y_axis_lims_from_default(
        ax, par_dict["y_scale"], (min_attenuation - y_off, y_off)
    )

    # Add fit peak identifiers
    first_corr = cut_xds.attrs["available_corrs"][0]

    _add_secondary_beam_hpbw_x_axis_to_plot(
        cut_xds.attrs[f"{first_corr}_pb_fwhm"] * lm_fac, ax
    )

    # Add bounded box with Beam parameters
    _add_beam_parameters_box(
        ax,
        pb_center * lm_fac,
        pb_fwhm * lm_fac,
        fsl_ratio,
        lm_unit,
        attenuation_plot=True,
    )
    return


def _plot_cuts_in_lm_sub(cut_xdtree, par_dict):
    """
    Produce plot of LM offsets in all cuts for antenna and ddi xd tree.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :return: None
    :rtype: NoneType
    """
    colors = ["blue", "red", "green", "black", "orange", "grey"]
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")

    fig, ax = create_figure_and_axes(None, [1, 1])
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        lm_offsets = lm_fac * cut_xds["lm_offsets"].values
        ax.plot(
            lm_offsets[:, 0],
            lm_offsets[:, 1],
            label=f'cut {icut}, {cut_xds.attrs["direction"]}',
            marker=".",
            ls="",
            color=colors[icut],
        )

    ax.legend(loc="best")
    ax.set_xlabel(f"L offset [{lm_unit}]")
    ax.set_ylabel(f"M offset [{lm_unit}]")
    ax.set_aspect("equal", adjustable="datalim")

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("lm_offsets", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


###########################################################
### Data labeling
###########################################################
def _make_parallel_hand_sub_title(attributes):
    """
    Make subtitle for data based on XDS attributes.

    :param attributes: beamcut xds attributes
    :type attributes: dict

    :return: Subtitle string
    :rtype: str
    """
    direction = attributes["direction"]
    time_string = attributes["time_string"]
    return f"{direction}, {time_string} UTC"


def _create_beamcut_header(summary, par_dict):
    """
    Create a data labeling header for plots and/or reports.

    :param summary: Data summary from xds attributes
    :type summary: dict

    :param par_dict: Parameter dictionary containing configuration parameters
    :type par_dict: dict

    :return: Data labeling header for plots and/or reports.
    :rtype: str
    """
    azel_unit = par_dict["azel_unit"]

    antenna = par_dict["this_ant"]
    ddi = par_dict["this_ddi"]
    freq_str = format_frequency(summary["spectral"]["rep. frequency"], decimal_places=3)
    raw_azel = np.array(summary["general"]["az el info"]["mean"])
    mean_azel = convert_unit("rad", azel_unit, "trigonometric") * raw_azel
    title = (
        f"Beam cut for {create_dataset_label(antenna, ddi, separator=',')}, "
        + r"$\nu$ = "
        + f"{freq_str}, "
    )
    if azel_unit == "rad":
        decimal_places = 3
    else:
        decimal_places = 1
    title += f"Az ~ {format_value_unit(mean_azel[0], azel_unit, decimal_places=decimal_places)}, "
    title += f"El ~ {format_value_unit(mean_azel[1], azel_unit, decimal_places=decimal_places)}"
    return title
