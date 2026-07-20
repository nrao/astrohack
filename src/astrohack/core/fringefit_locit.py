import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, CIRS, AltAz
from casacoretables import tables as ctables
from astropy.time import Time
import astropy.units as u

from astrohack.utils.algorithms import least_squares
from astrohack.utils.constants import clight
import toolviper.utils.logger as logger

from astrohack.utils.conversion import convert_unit
from astrohack.utils.text import fixed_format_error, create_pretty_table


def _fetch_field_info(fringefit_caltable):
    field_table = ctables.table(
        f"{fringefit_caltable}/FIELD",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    source_ids = field_table.getcol("SOURCE_ID")
    phase_dir = field_table.getcol("PHASE_DIR")
    field_names = field_table.getcol("NAME")
    field_table.close()

    obs_table = ctables.table(
        fringefit_caltable + "/OBSERVATION",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    time_range = obs_table.getcol("TIME_RANGE")[0] / 86400
    telescope_name = obs_table.getcol("TELESCOPE_NAME")[0]
    obs_table.close()

    field_dict = {"time_range": time_range, "telescope_name": telescope_name}
    for idx, source_id in enumerate(source_ids):
        src_dict = {
            "id": source_id,
            "name": field_names[idx],
            "J2000": phase_dir[idx],
        }
        field_dict[str(source_id)] = src_dict

    return field_dict


def _extract_antenna_info(fringefit_caltable):
    ant_table = ctables.table(
        f"{fringefit_caltable}/ANTENNA",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    ant_names = ant_table.getcol("NAME")
    ant_pos = ant_table.getcol("POSITION")
    ant_station = ant_table.getcol("STATION")
    ant_table.close()

    ant_dict = {}

    for ant_id, ant_name in enumerate(ant_names):
        ant_dict[str(ant_id)] = {
            "name": ant_name,
            "position": ant_pos[ant_id],
            "station": ant_station[ant_id],
        }

    return ant_dict


def _extract_delay_info(fringefit_caltable):
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
    field = main_table.getcol("FIELD_ID")

    delay_dict = {
        "time": times,
        "ant1": ant1,
        "ant2": ant2,
        "delays": fparam[:, 0, 1::4] * 1e-9,  # convert to sec
        "field": field,
    }

    # print(main_table.colnames())

    return delay_dict


def _match_delays_to_ha_and_dec(field_dict, ant_dict, delay_dict, pol_sel):
    unq_ants = np.unique(delay_dict["ant1"])
    unq_refants, refant_count = np.unique(delay_dict["ant2"], return_counts=True)
    refant_id = unq_refants[0]
    init_time = field_dict["time_range"][0]

    matched_dict = {}
    for ant_id in unq_ants:
        ant_info = ant_dict[str(ant_id)]
        ant_name = ant_info["name"]
        ant_sel = np.logical_and(
            delay_dict["ant1"] == ant_id, delay_dict["ant2"] == refant_id
        )
        sel_delay = delay_dict["delays"][ant_sel]
        if np.sum(ant_sel) == 0 or np.sum(sel_delay) == 0:
            logger.warning(f"No delays for antenna {ant_name}")
            continue
        logger.info(f"Matching sky coords to delays for {ant_name}")

        geo_pos = ant_info["position"]
        ant_location = EarthLocation.from_geocentric(
            geo_pos[0],
            geo_pos[1],
            geo_pos[2],
            "meter",
        )
        sel_time = delay_dict["time"][ant_sel]
        sel_field = delay_dict["field"][ant_sel]
        j2000_radec = np.zeros_like(sel_delay)
        for row, atime in enumerate(sel_time):
            j2000_radec[row, :] = field_dict[str(sel_field[row])]["J2000"]
        ant_times = Time(
            sel_time / 86400, format="mjd", scale="utc", location=ant_location
        )
        skycoords = SkyCoord(
            ra=j2000_radec[:, 0] * u.rad, dec=j2000_radec[:, 1] * u.rad, frame="icrs"
        ).transform_to(CIRS(obstime=ant_times))
        lst = ant_times.sidereal_time("apparent").to(u.rad) / u.rad
        ra = skycoords.ra.rad
        hour_angle = lst - ra
        altaz_frame = AltAz(location=ant_location, obstime=ant_times)
        altaz_coords = skycoords.transform_to(altaz_frame)
        n_rows = sel_time.shape[0]
        n_pol = len(pol_sel)
        coordinate_array = np.zeros((4, n_pol * n_rows))
        delay_array = np.zeros(n_pol * n_rows)
        for i_pol in range(n_pol):
            f_row = i_pol * n_rows
            l_row = (i_pol + 1) * n_rows
            coordinate_array[0, f_row:l_row] = hour_angle.value
            coordinate_array[1, f_row:l_row] = skycoords.dec.rad
            coordinate_array[2, f_row:l_row] = altaz_coords.alt.rad
            coordinate_array[3, f_row:l_row] = sel_time - init_time
            delay_array[f_row:l_row] = sel_delay[:, i_pol]
        this_ant_data = {
            "station": ant_info["station"],
            "coordinates": coordinate_array,
            "delays": delay_array,
        }

        matched_dict[ant_name] = this_ant_data

        pass

    return matched_dict


def _fit_one_ant(ant_name, ant_data, fit_delay_rate, fit_kterm, pol_sel):
    from astrohack.core.locit import _system_coefficients

    logger.info(f"Fitting antenna {ant_name}")
    delays = ant_data["delays"]
    elevation = ant_data["elevation"]
    hour_angle = ant_data["hour_angle"]
    dec = ant_data["dec"]
    times = ant_data["time"]

    npol = len(pol_sel)
    npar = 4 + fit_delay_rate + fit_kterm
    nrows = delays.shape[0] * npol

    system = np.zeros([nrows, npar])
    vector = np.zeros(nrows)

    for row, el_val in enumerate(elevation):
        for pol in pol_sel:
            coords = [hour_angle[row], dec[row], el_val, times[row]]
            system_row = npol * row + pol
            system[system_row] = _system_coefficients(coords, fit_delay_rate, fit_kterm)
            vector[system_row] = delays[row, pol]

    fit, variance, _ = least_squares(system, vector)
    predictions = system @ fit
    delay_rms = np.sqrt(np.mean((predictions - vector) ** 2))

    return fit, variance, delay_rms


def _process_all_ant(matched_dict, fit_delay_rate, fit_kterm, pol_sel, fit_engine):
    from astrohack.core.locit import (
        _solve_linear_algebra,
        _solve_scipy_optimize_curve_fit,
        _compute_chi_squared,
    )

    if fit_engine == "linear algebra":
        fit_func = _solve_linear_algebra
    else:
        fit_func = _solve_scipy_optimize_curve_fit

    result_dict = {}
    for ant_name, ant_data in matched_dict.items():
        fit, variance = fit_func(
            ant_data["coordinates"], ant_data["delays"], fit_kterm, fit_delay_rate
        )
        model, chi2 = _compute_chi_squared(
            ant_data["delays"], fit, ant_data["coordinates"], fit_kterm, fit_delay_rate
        )
        result_dict[ant_name] = {
            "data": ant_data,
            "fit": {"values": fit, "errors": variance, "chi2": chi2, "model": model},
        }

    return result_dict


def _get_formated_row(ant_name, fit, fit_error, delay_rms, fit_kterm, fit_delay_rate):
    pos_fact = clight
    delay_fact = 1e9
    row = [ant_name, f"{delay_rms * delay_fact:.2e}"]

    factors = [delay_fact, pos_fact, pos_fact, pos_fact]
    if fit_kterm:
        factors.append(pos_fact)
    if fit_delay_rate:
        factors.append(delay_fact * 3600)

    for i_val, value in enumerate(fit):
        row.append(
            fixed_format_error(
                value, fit_error[i_val], factors[i_val], 1e-2  # * factors[i_val]
            )
        )

    return row


def _plot_delays_with_model(ant_name, coordinates, delays, model, delay_lims):
    from astrohack.visualization.plot_tools import (
        create_figure_and_axes,
        close_figure,
        scatter_plot,
    )

    delays = delays * 1e9
    model = model * 1e9

    rad2deg = convert_unit("rad", "deg", "trigonometric")
    fig, axes = create_figure_and_axes(None, [2, 2])
    ylabel = "Delays [nsec]"
    scatter_plot(
        axes[0, 0],
        coordinates[3, :],
        f"Time from observation start [s]",
        delays,
        ylabel,
        "Time vs Delays",
        ylim=delay_lims,
        model=model,
    )
    scatter_plot(
        axes[0, 1],
        coordinates[2, :] * rad2deg,
        f"Elevation [deg]",
        delays,
        ylabel,
        "Elevation vs Delays",
        xlim=[-5, 95],
        ylim=delay_lims,
        model=model,
    )
    scatter_plot(
        axes[1, 0],
        coordinates[0, :] * rad2deg,
        f"Hour Angle [deg]",
        delays,
        ylabel,
        "Hour Angle vs Delays",
        ylim=delay_lims,
        model=model,
    )
    scatter_plot(
        axes[1, 1],
        coordinates[1, :] * rad2deg,
        f"Declination [deg]",
        delays,
        ylabel,
        "Declination vs Delays",
        xlim=[-95, 95],
        ylim=delay_lims,
        model=model,
    )
    close_figure(
        fig,
        f"Delay fit results for {ant_name}",
        f"fringefit_locit_{ant_name}.png",
        300,
        False,
        True,
    )


def _post_process_all_ant(
    fringefit_caltable, result_dict, fit_delay_rate, fit_kterm, delay_lims
):
    field_names = [
        "Antenna",
        "Fit RMS [nsec]",
        "Fixed delay [nsec]",
        "X [m]",
        "Y [m]",
        "Z [m]",
    ]
    if fit_kterm:
        field_names += ["K-term [m]"]
    if fit_delay_rate:
        field_names += ["Delay rate [nsec/hr]"]

    table = create_pretty_table(field_names)
    for ant_name, ant_data in result_dict.items():
        fit_val = ant_data["fit"]["values"]
        fit_err = ant_data["fit"]["errors"]
        fit_chi2 = ant_data["fit"]["chi2"]
        row = _get_formated_row(
            ant_name, fit_val, fit_err, fit_chi2, fit_kterm, fit_delay_rate
        )
        _plot_delays_with_model(
            ant_name,
            ant_data["data"]["coordinates"],
            ant_data["data"]["delays"],
            ant_data["fit"]["model"],
            delay_lims,
        )
        table.add_row(row)

    print(f"Results for {fringefit_caltable}:")
    print(table.get_string())
    return


def fringefit_locit(
    fringefit_caltable,
    fit_delay_rate=True,
    fit_kterm=True,
    pol_sel=[0, 1],
    fit_engine="scipy",
    delay_lims=[-100, 100],
):

    field_dict = _fetch_field_info(fringefit_caltable)
    ant_dict = _extract_antenna_info(fringefit_caltable)

    delay_dict = _extract_delay_info(fringefit_caltable)
    matched_dict = _match_delays_to_ha_and_dec(
        field_dict, ant_dict, delay_dict, pol_sel
    )

    result_dict = _process_all_ant(
        matched_dict, fit_delay_rate, fit_kterm, pol_sel, fit_engine
    )

    _post_process_all_ant(
        fringefit_caltable, result_dict, fit_delay_rate, fit_kterm, delay_lims
    )
