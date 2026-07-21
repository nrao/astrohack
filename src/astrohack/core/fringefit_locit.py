import shutil

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, CIRS, AltAz
from casacoretables import tables as ctables
from astropy.time import Time
import astropy.units as u

from astrohack import AstrohackPositionFile
import toolviper.utils.logger as logger

from astrohack.utils.conversion import convert_unit


def fringefit_locit_looping_dict(
    locit_parms: dict, full_antenna_list: list
) -> tuple[dict, str]:
    fringefit_caltable = locit_parms["fringefit_caltable"]
    main_table = ctables.table(
        f"{fringefit_caltable}",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    times = main_table.getcol("TIME")
    ant1 = main_table.getcol("ANTENNA1")
    ant2 = main_table.getcol("ANTENNA2")
    fparam = main_table.getcol("FPARAM")
    spw = main_table.getcol("SPECTRAL_WINDOW_ID")
    field = main_table.getcol("FIELD_ID")

    delays = fparam[:, 0, 1::4] * 1e-9
    if locit_parms["ant"] == "all":
        looping_ant_list = full_antenna_list
    else:
        if isinstance(locit_parms["ant"], str):
            user_ant_list = [locit_parms["ant"]]
        else:
            user_ant_list = locit_parms["ant"]
        looping_ant_list = []
        for ant_name in user_ant_list:
            if ant_name in full_antenna_list:
                looping_ant_list.append(ant_name)

    unq_refants = np.unique(ant2)
    refant_id = unq_refants[0]
    if unq_refants.size != 1:
        logger.warning(
            "More than one refant not supported dropping data with alternative reference antennas"
        )
    ant2_sel = ant2 == refant_id
    refant_name = full_antenna_list[refant_id]
    looping_dict = {}
    unq_ants_in_data = np.unique(ant1)
    for ant_id in unq_ants_in_data:
        ant_name = full_antenna_list[ant_id]
        if ant_name not in looping_ant_list:
            continue
        ant_key = f"ant_{ant_name}"
        ant_selection = np.logical_and(ant1 == ant_id, ant2_sel)

        if np.sum(delays[ant_selection]) > 0:
            this_ant_data = {
                "time": times[ant_selection],
                "delays": delays[ant_selection],  # convert to sec
                "fields": field[ant_selection],
                "spw": spw[ant_selection],
            }
            looping_dict[ant_key] = this_ant_data
        else:
            logger.warning(f"No valid delay data for {ant_name}")
            shutil.rmtree(f"{locit_parms['position_name']}/{ant_key}")
    return looping_dict, refant_name


def _match_delays_to_coordinates(
    locit_parms: dict,
    field_dict: dict,
    ant_info: dict,
    delay_dict: dict,
    init_time: float,
    ddi_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list]:
    user_pol_sel = locit_parms["polarization"]
    el_limit = (
        convert_unit("deg", "rad", "trigonometric") * locit_parms["elevation_limit"]
    )

    if user_pol_sel == "both":
        pol_sel = [0, 1]
    elif user_pol_sel == "R":
        pol_sel = [0]
    elif user_pol_sel == "L":
        pol_sel = [1]
    else:
        raise ValueError(f"Polarization selection ({user_pol_sel}) not recognized")

    spw = delay_dict["spw"]
    ddi_sel = np.full_like(spw, False)
    used_ddis = []
    for ddi in ddi_dict.keys():
        this_ddi_sel = spw == ddi
        if np.sum(this_ddi_sel) > 0:
            used_ddis.append(ddi)
        ddi_sel = np.logical_or(this_ddi_sel, ddi_sel)

    ant_time = delay_dict["time"][ddi_sel]
    ant_fields = delay_dict["fields"][ddi_sel]
    ant_delays = delay_dict["delays"][ddi_sel]

    geo_pos = ant_info["geocentric_position"]
    ant_location = EarthLocation.from_geocentric(
        geo_pos[0],
        geo_pos[1],
        geo_pos[2],
        "meter",
    )
    j2000_radec = np.zeros_like(ant_delays)
    for row, atime in enumerate(ant_time):
        j2000_radec[row, :] = field_dict[ant_fields[row]]["fk5"]
    ant_times = Time(ant_time / 86400, format="mjd", scale="utc", location=ant_location)
    skycoords = SkyCoord(
        ra=j2000_radec[:, 0] * u.rad, dec=j2000_radec[:, 1] * u.rad, frame="icrs"
    ).transform_to(CIRS(obstime=ant_times))
    lst = ant_times.sidereal_time("apparent").to(u.rad) / u.rad
    ra = skycoords.ra.rad
    hour_angle = lst - ra
    altaz_frame = AltAz(location=ant_location, obstime=ant_times)
    altaz_coords = skycoords.transform_to(altaz_frame)
    n_rows = ant_time.shape[0]
    n_pol = len(pol_sel)
    coordinate_array = np.zeros((4, n_pol * n_rows))
    delay_array = np.zeros(n_pol * n_rows)
    lst_array = np.zeros(n_pol * n_rows)
    for i_pol in pol_sel:
        f_row = i_pol * n_rows
        l_row = (i_pol + 1) * n_rows
        coordinate_array[0, f_row:l_row] = hour_angle.value
        coordinate_array[1, f_row:l_row] = skycoords.dec.rad
        coordinate_array[2, f_row:l_row] = altaz_coords.alt.rad
        coordinate_array[3, f_row:l_row] = ant_time - init_time
        delay_array[f_row:l_row] = ant_delays[:, i_pol]
        lst_array[f_row:l_row] = lst

    el_selection = coordinate_array[2, :] >= el_limit
    return (
        coordinate_array[:, el_selection],
        delay_array[el_selection],
        lst_array,
        el_limit,
        used_ddis,
    )


def _get_average_freq(ddi_dict: dict, used_ddis: list) -> float:
    freqs = []
    bws = []
    for key, value in ddi_dict.items():
        if key in used_ddis:
            freqs.append(value["frequency"])
            bws.append(value["bandwidth"][0])
    average_freq = float(np.average(freqs, weights=bws))
    return average_freq


def fringefit_locit_chunk(locit_parms: dict, output_mds: AstrohackPositionFile):
    from astrohack.core.locit import (
        _solve_linear_algebra,
        _solve_scipy_optimize_curve_fit,
        _compute_chi_squared,
        _create_output_xds,
    )

    ddi_dict = locit_parms["ddi_dict"]
    ant_order = [locit_parms["this_ant"]]
    current_xds = output_mds.open_subset(ant_order)
    antenna_info = current_xds.attrs["antenna_info"]
    src_dict = output_mds.root.attrs["source_dict"]
    ant_name = antenna_info["name"]
    logger.info(f"Processing {ant_name}")

    delay_dict = locit_parms["dic_data"]
    init_time = output_mds.root.attrs["time_range"][0]
    coordinates, delays, lst, el_limit, used_ddis = _match_delays_to_coordinates(
        locit_parms, src_dict, antenna_info, delay_dict, init_time, ddi_dict
    )
    if coordinates.size == 0:
        logger.warning(f"Data selection excludes all data for {ant_name}")
        shutil.rmtree(f"{output_mds.filename}/{locit_parms['this_ant']}")
        return

    average_freq = _get_average_freq(ddi_dict, used_ddis)
    fit_kterm = locit_parms["fit_kterm"]
    fit_delay_rate = locit_parms["fit_delay_rate"]
    if locit_parms["fit_engine"] == "linear algebra":
        fit_func = _solve_linear_algebra
    else:
        fit_func = _solve_scipy_optimize_curve_fit

    fit, variance = fit_func(coordinates, delays, fit_kterm, fit_delay_rate)
    model, chi2 = _compute_chi_squared(
        delays, fit, coordinates, fit_kterm, fit_delay_rate
    )

    current_xds = _create_output_xds(
        coordinates,
        lst,
        delays,
        fit,
        variance,
        chi2,
        model,
        locit_parms,
        average_freq,
        el_limit,
        antenna_info,
    )
    output_mds.add_node(current_xds, ant_order)
    return
