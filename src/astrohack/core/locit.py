from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt

import toolviper.utils.logger as logger
import astropy.units as units
import xarray as xr

from astrohack.utils import (
    get_data_name,
    create_dataset_label,
    fixed_format_error,
    rotate_to_gmt,
    compute_antenna_relative_off,
)

from astrohack.visualization.diagnostics import plot_one_antenna_position
from astrohack.utils.conversion import convert_unit, hadec_to_elevation
from astrohack.utils.algorithms import least_squares, phase_wrapping
from astrohack.utils.constants import *
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.visualization import (
    create_figure_and_axes,
    scatter_plot,
    close_figure,
    plot_boxes_limits_and_labels,
)


def locit_separated_chunk(locit_parms, output_mds):
    """
    This is the chunk function for locit when treating each DDI separately
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    Returns:
    xds save to disk in the .zarr format
    """
    input_xdt = locit_parms["xdt_data"]
    field_id, time, delays, freq = _get_data_from_locit_xds(
        input_xdt, locit_parms["polarization"]
    )
    ant_key = locit_parms["this_ant"]
    ddi_key = locit_parms["this_ddi"]
    antenna_info = input_xdt.parent.attrs["antenna_info"]
    source_dict = input_xdt.parent.parent.attrs["source_dict"]
    if _has_valid_data(field_id, time, delays, ant_key, ddi=ddi_key):

        coordinates, delays, lst, elevation_limit, nin = _build_filtered_arrays(
            field_id, time, delays, locit_parms, antenna_info, source_dict
        )
        if _elevation_ok(nin, locit_parms["this_ant"]):
            fit, variance, converged = _fit_data(coordinates, delays, locit_parms)
            if converged:
                model, chi_squared = _compute_chi_squared(
                    delays,
                    fit,
                    coordinates,
                    locit_parms["fit_kterm"],
                    locit_parms["fit_delay_rate"],
                )
                new_node = _create_output_xds(
                    coordinates,
                    lst,
                    delays,
                    fit,
                    variance,
                    chi_squared,
                    model,
                    locit_parms,
                    freq,
                    elevation_limit,
                    antenna_info,
                    ant_key,
                    ddi_key,
                )
                output_mds.add_node_to_tree(
                    new_node,
                    dump_to_disk=False,
                    running_in_parallel=locit_parms["parallel"],
                )


def locit_combined_chunk(locit_parms, output_mds):
    """
    This is the chunk function for locit when we are combining the DDIs for an antenna for a single solution
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    Returns:
    xds save to disk in the .zarr format
    """
    ant_xdt = locit_parms["xdt_data"]
    antenna_info = ant_xdt.attrs["antenna_info"]
    source_dict = ant_xdt.parent.attrs["source_dict"]
    ant_key = locit_parms["this_ant"]

    delay_list = []
    time_list = []
    field_list = []
    freq_list = []

    for ddi, xdt_data in ant_xdt.items():
        this_field_id, this_time, this_delays, freq = _get_data_from_locit_xds(
            xdt_data, locit_parms["polarization"]
        )
        freq_list.append(freq)
        field_list.append(this_field_id)
        time_list.append(this_time)
        delay_list.append(this_delays)

    delays = np.concatenate(delay_list)
    time = np.concatenate(time_list)
    field_id = np.concatenate(field_list)

    if _has_valid_data(field_id, time, delays, locit_parms["this_ant"]):
        coordinates, delays, lst, elevation_limit, nin = _build_filtered_arrays(
            field_id, time, delays, locit_parms, antenna_info, source_dict
        )
        if _elevation_ok(nin, locit_parms["this_ant"]):
            fit, variance, converged = _fit_data(coordinates, delays, locit_parms)
            if converged:
                model, chi_squared = _compute_chi_squared(
                    delays,
                    fit,
                    coordinates,
                    locit_parms["fit_kterm"],
                    locit_parms["fit_delay_rate"],
                )
                new_node = _create_output_xds(
                    coordinates,
                    lst,
                    delays,
                    fit,
                    variance,
                    chi_squared,
                    model,
                    locit_parms,
                    freq_list,
                    elevation_limit,
                    antenna_info,
                    ant_key,
                )
                output_mds.add_node_to_tree(
                    new_node,
                    dump_to_disk=True,
                    running_in_parallel=locit_parms["parallel"],
                )


def locit_difference_chunk(locit_parms, output_mds):
    """
    This is the chunk function for locit when we are combining two DDIs for an antenna for a single solution by using
    the difference in phase between the two DDIs of different frequencies
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    """
    ant_xdt = locit_parms["xdt_data"]
    antenna_info = ant_xdt.attrs["antenna_info"]
    source_dict = ant_xdt.parent.attrs["source_dict"]
    ant_key = locit_parms["this_ant"]

    ddi_list = list(ant_xdt.keys())
    nddis = len(ddi_list)

    if nddis != 2:
        msg = f"The difference method support only 2 DDIs, {nddis} DDIs provided for Antenna {ant_key.split('_')[1]}."
        logger.error(msg)
        return None

    ddi_0 = _get_data_from_locit_xds(
        ant_xdt[ddi_list[0]],
        locit_parms["polarization"],
        get_phases=True,
        split_pols=True,
    )
    ddi_1 = _get_data_from_locit_xds(
        ant_xdt[ddi_list[1]],
        locit_parms["polarization"],
        get_phases=True,
        split_pols=True,
    )

    time, field_id, delays, freq = _delays_from_phase_differences(ddi_0, ddi_1)
    if _has_valid_data(field_id, time, delays, locit_parms["this_ant"]):
        coordinates, delays, lst, elevation_limit, nin = _build_filtered_arrays(
            field_id, time, delays, locit_parms, antenna_info, source_dict
        )
        if _elevation_ok(nin, locit_parms["this_ant"]):
            fit, variance, converged = _fit_data(coordinates, delays, locit_parms)
            if converged:
                model, chi_squared = _compute_chi_squared(
                    delays,
                    fit,
                    coordinates,
                    locit_parms["fit_kterm"],
                    locit_parms["fit_delay_rate"],
                )
                new_node = _create_output_xds(
                    coordinates,
                    lst,
                    delays,
                    fit,
                    variance,
                    chi_squared,
                    model,
                    locit_parms,
                    freq,
                    elevation_limit,
                    antenna_info,
                    ant_key,
                )
                output_mds.add_node_to_tree(
                    new_node,
                    dump_to_disk=True,
                    running_in_parallel=locit_parms["parallel"],
                )


def plot_sky_coverage_chunk(parm_dict):
    """
    Plot the sky coverage for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the sky coverage
    """

    ant_xdt = parm_dict["xdt_data"]
    combined = parm_dict["combined"]
    antenna = parm_dict["this_ant"]
    destination = parm_dict["destination"]

    if combined:
        export_name = f"{destination}/position_sky_coverage_{antenna}.png"
        suptitle = f'Sky coverage for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_sky_coverage_{antenna}_{ddi}.png"
        suptitle = (
            f'Sky coverage for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'
        )

    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = ant_xdt.attrs["antenna_info"]

    time = ant_xdt.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    ha = ant_xdt["HOUR_ANGLE"] * angle_fact
    dec = ant_xdt["DECLINATION"] * angle_fact
    ele = ant_xdt["ELEVATION"] * angle_fact

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    elelim, elelines, declim, declines, halim = _compute_plot_borders(
        angle_fact, antenna_info["latitude"], ant_xdt.attrs["elevation_limit"]
    )
    timelabel = f"Time from observation start [{time_unit}]"
    halabel = f"Hour Angle [{angle_unit}]"
    declabel = f"Declination [{angle_unit}]"
    scatter_plot(
        axes[0, 0],
        time,
        timelabel,
        ele,
        f"Elevation [{angle_unit}]",
        "Time vs Elevation",
        ylim=elelim,
        hlines=elelines,
        add_legend=False,
    )
    scatter_plot(
        axes[0, 1],
        time,
        timelabel,
        ha,
        halabel,
        "Time vs Hour angle",
        ylim=halim,
        add_legend=False,
    )
    scatter_plot(
        axes[1, 0],
        time,
        timelabel,
        dec,
        declabel,
        "Time vs Declination",
        ylim=declim,
        hlines=declines,
        add_legend=False,
    )
    scatter_plot(
        axes[1, 1],
        ha,
        halabel,
        dec,
        declabel,
        "Hour angle vs Declination",
        ylim=declim,
        xlim=halim,
        hlines=declines,
        add_legend=False,
    )

    close_figure(fig, suptitle, export_name, dpi, display)
    return


def plot_delays_chunk(parm_dict):
    """
    Plot the delays and optionally the delay model for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the delay plots
    """
    combined = parm_dict["combined"]
    plot_model = parm_dict["plot_model"]
    antenna = parm_dict["this_ant"]
    destination = parm_dict["destination"]
    if combined:
        export_name = f'{destination}/position_delays_{antenna}_combined_{parm_dict["comb_type"]}.png'
        suptitle = f'Delays for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_delays_{antenna}_separated_{ddi}.png"
        suptitle = (
            f'Delays for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'
        )

    ant_xdt = parm_dict["xdt_data"]
    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    delay_unit = parm_dict["delay_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = ant_xdt.attrs["antenna_info"]

    time = ant_xdt.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    delay_fact = convert_unit("sec", delay_unit, kind="time")
    ha = ant_xdt["HOUR_ANGLE"] * angle_fact
    dec = ant_xdt["DECLINATION"] * angle_fact
    ele = ant_xdt["ELEVATION"] * angle_fact
    delays = ant_xdt["DELAYS"].values * delay_fact

    elelim, elelines, declim, declines, halim = _compute_plot_borders(
        angle_fact, antenna_info["latitude"], ant_xdt.attrs["elevation_limit"]
    )
    delay_minmax = [np.min(delays), np.max(delays)]
    delay_border = 0.05 * (delay_minmax[1] - delay_minmax[0])
    delaylim = [delay_minmax[0] - delay_border, delay_minmax[1] + delay_border]

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    ylabel = f"Delays [{delay_unit}]"
    if plot_model:
        model = ant_xdt["MODEL"].values * delay_fact
    else:
        model = None
    scatter_plot(
        axes[0, 0],
        time,
        f"Time from observation start [{time_unit}]",
        delays,
        ylabel,
        "Time vs Delays",
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[0, 1],
        ele,
        f"Elevation [{angle_unit}]",
        delays,
        ylabel,
        "Elevation vs Delays",
        xlim=elelim,
        vlines=elelines,
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[1, 0],
        ha,
        f"Hour Angle [{angle_unit}]",
        delays,
        ylabel,
        "Hour Angle vs Delays",
        xlim=halim,
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[1, 1],
        dec,
        f"Declination [{angle_unit}]",
        delays,
        ylabel,
        "Declination vs Delays",
        xlim=declim,
        vlines=declines,
        ylim=delaylim,
        model=model,
    )

    close_figure(fig, suptitle, export_name, dpi, display)
    return


def _delays_from_phase_differences(ddi_0, ddi_1):
    """
    Compute delays from the difference in phase between two DDIs of different frequencies
    Args:
        ddi_0: First DDI
        ddi_1: Second DDI

    Returns:
    Matched times, matched field ids, matched phase difference delays, difference in frequency
    """

    freq = ddi_0[3] - ddi_1[3]
    if freq > 0:
        pos_time, pos_phase = ddi_0[1:3]
        neg_time, neg_phase = ddi_1[1:3]
        fields = ddi_0[0]
    elif freq < 0:
        pos_time, pos_phase = ddi_1[1:3]
        neg_time, neg_phase = ddi_0[1:3]
        freq *= -1
        fields = ddi_1[0]
    else:
        msg = f"The two DDIs must have different frequencies"
        logger.error(msg)
        raise Exception(msg)

    if isinstance(fields, list):
        time = []
        field_id = []
        phase = []
        for i_pol in range(len(fields)):
            this_time, this_field_id, this_phase = _match_times_and_phase_difference(
                pos_time[i_pol],
                neg_time[i_pol],
                pos_phase[i_pol],
                neg_phase[i_pol],
                fields[i_pol],
            )
            time.append(this_time)
            field_id.append(this_field_id)
            phase.append(this_phase)

        time = np.concatenate(time)
        field_id = np.concatenate(field_id)
        phase = np.concatenate(phase)

    else:
        time, field_id, phase = _match_times_and_phase_difference(
            pos_time, neg_time, pos_phase, neg_phase, fields
        )

    delays = phase / twopi / freq
    return time, field_id, delays, freq


def _match_times_and_phase_difference(
    pos_time, neg_time, pos_phase, neg_phase, fields, tolerance=1e-8
):
    """
    match times and compute the phase differences for the simple case, calls _different_times for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference
    """
    n_pos_time, n_neg_time = len(pos_time), len(neg_time)
    if n_pos_time == n_neg_time:
        if np.all(
            np.isclose(pos_time, neg_time, tolerance)
        ):  # this the simplest case times are already matched!
            return pos_time, fields, phase_wrapping(pos_phase - neg_phase)
        else:
            return _different_times(
                pos_time, neg_time, pos_phase, neg_phase, fields, tolerance
            )
    else:
        return _different_times(
            pos_time, neg_time, pos_phase, neg_phase, fields, tolerance
        )


def _different_times(pos_time, neg_time, pos_phase, neg_phase, fields, tolerance=1e-8):
    """
    match times and compute the phase differences for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference
    """
    # This solution is not optimal but numpy does not have a task for it, if it ever becomes a bottleneck we can JIT it
    out_times = np.sort(
        [time for time in pos_time if np.isclose(neg_time, time, tolerance).any()]
    )
    ntimes = out_times.shape[0]
    out_phase = np.ndarray(ntimes)
    out_field = np.ndarray(ntimes, dtype=np.int64)

    for i_time in range(ntimes):
        i_pos = np.absolute(pos_time - out_times[i_time]).argmin()
        i_neg = np.absolute(neg_time - out_times[i_time]).argmin()
        out_phase[i_time] = pos_phase[i_pos] - neg_phase[i_neg]
        out_field[i_time] = fields[i_pos]

    return out_times, out_field, phase_wrapping(out_phase)


def _has_valid_data(field_id, time, delays, antenna, ddi=None):
    """
    Determine if locit xds has valid data for locit purposes
    :param field_id: Array of field ids in time.
    :param time: Time axis.
    :param delays: Array of delays  in time
    :param antenna: Antenna key
    :param ddi: DDI key
    :return:
    """
    msg = f"Antenna {get_data_name(antenna)} "
    if ddi is not None:
        msg += f"DDI {get_data_name(ddi)} "
    msg += "has no valid data"
    if len(field_id) == 0 or len(time) == 0 or len(delays) == 0:
        logger.warning(msg)
        return False
    else:
        return True


def _elevation_ok(nin, antenna, ddi=None):
    """
    Determine if elevation limit takes out all the data.
    :param nin: Number of filtered points
    :param antenna: antenna key
    :param ddi: ddi key
    :return: True or False
    """
    msg = f"Antenna {get_data_name(antenna)} "
    if ddi is not None:
        msg += f"DDI {get_data_name(ddi)} "
    msg += "has no valid data, try decreasing the elevation limit."
    if nin > 0:
        return True
    else:
        logger.warning(msg)
        return False


def _get_data_from_locit_xds(
    xds_data, pol_selection, get_phases=False, split_pols=False
):
    """
    Extract data from a .locit.zarr xds, converts the phase gains to delays using the xds frequency
    Args:
        xds_data: The .locit.zarr xds
        pol_selection: Which polarization is requested from the xds
        get_phases: return phases rather than delays
        split_pols: Different polarizations are not concatenated in a single array if True


    Returns:
        the field ids
        the time in mjd
        The delays in seconds or phases in radians
        Xds frequency

    """

    pol = xds_data.attrs["polarization_scheme"]
    freq = xds_data.attrs["frequency"]

    if len(pol) != 2:
        msg = (
            f"Polarization scheme {pol} is not what is expected for antenna based gains"
        )
        logger.error(msg)
        raise Exception(msg)
    elif pol_selection == "both":
        phases = [
            xds_data[f"P0_PHASE_GAINS"].values,
            xds_data[f"P1_PHASE_GAINS"].values,
        ]
        field_id = [xds_data[f"P0_FIELD_ID"].values, xds_data[f"P1_FIELD_ID"].values]
        time = [xds_data.p0_time.values, xds_data.p1_time.values]
        if not split_pols:
            phases = np.concatenate(phases)
            field_id = np.concatenate(field_id)
            time = np.concatenate(time)
    else:
        sel_pol_list = [*pol_selection]
        phases = []
        time = []
        field_id = []
        for pol_item in sel_pol_list:
            if pol_item in pol:
                i_pol = np.where(np.array(pol) == pol_item)[0][0]
                phases.append(xds_data[f"P{i_pol}_PHASE_GAINS"].values)
                time.append(xds_data[f"p{i_pol}_time"].values)
                field_id.append(xds_data[f"P{i_pol}_FIELD_ID"].values)
            else:
                msg = f"Polarization {pol_selection} is not found in data"
                logger.warning(msg)

        if len(phases) == 0:
            msg = f"No valid data found for polarization selection {pol_selection}"
            logger.error(msg)
            raise Exception(msg)

        if not split_pols:
            phases = np.concatenate(phases)
            field_id = np.concatenate(field_id)
            time = np.concatenate(time)

    if get_phases:
        return field_id, time, phases, freq  # field_id, time, phases, frequency
    else:
        return (
            field_id,
            time,
            phases / twopi / freq,
            freq,
        )  # field_id, time, delays, frequency


def _create_output_xds(
    coordinates,
    lst,
    delays,
    fit,
    variance,
    chi_squared,
    model,
    locit_parms,
    frequency,
    elevation_limit,
    antenna_info,
    ant_key,
    ddi_key=None,
):
    """
    Create the output xds from the computed quantities and the fit results
    Args:
        coordinates: The coordinate array used in the fitting
        lst: The local sidereal time
        delays: The fitted delays
        fit: The fit results
        variance: the fit error bars
        locit_parms: the input parameters
        frequency: The frequency or frequencies of the input xds or xdses
        elevation_limit: the elevation cutoff

    Returns:
    The xdt to be plugged to root.
    """
    fit_kterm = locit_parms["fit_kterm"]
    fit_rate = locit_parms["fit_delay_rate"]
    error = np.sqrt(variance)

    # print(delays)

    output_xds = xr.Dataset()
    output_xds.attrs["polarization"] = locit_parms["polarization"]
    output_xds.attrs["frequency"] = frequency
    output_xds.attrs["position_fit"] = fit[1:4]
    output_xds.attrs["position_error"] = error[1:4]
    output_xds.attrs["fixed_delay_fit"] = fit[0]
    output_xds.attrs["fixed_delay_error"] = error[0]
    output_xds.attrs["antenna_info"] = antenna_info
    output_xds.attrs["elevation_limit"] = elevation_limit
    output_xds.attrs["chi_squared"] = chi_squared

    if fit_kterm and fit_rate:
        output_xds.attrs["koff_fit"] = fit[4]
        output_xds.attrs["koff_error"] = error[4]
        output_xds.attrs["rate_fit"] = fit[5]
        output_xds.attrs["rate_error"] = error[5]
    elif fit_kterm and not fit_rate:
        output_xds.attrs["koff_fit"] = fit[4]
        output_xds.attrs["koff_error"] = error[4]
    elif not fit_kterm and fit_rate:
        output_xds.attrs["rate_fit"] = fit[4]
        output_xds.attrs["rate_error"] = error[4]
    else:
        pass  # Nothing to be added to the attributes

    coords = {"time": coordinates[3, :]}
    output_xds["DELAYS"] = xr.DataArray(delays, dims=["time"])
    output_xds["MODEL"] = xr.DataArray(model, dims=["time"])
    output_xds["HOUR_ANGLE"] = xr.DataArray(coordinates[0, :], dims=["time"])
    output_xds["DECLINATION"] = xr.DataArray(coordinates[1, :], dims=["time"])
    output_xds["ELEVATION"] = xr.DataArray(coordinates[2, :], dims=["time"])
    output_xds["LST"] = xr.DataArray(lst, dims=["time"])

    # print(output_xds["DELAYS"].values)

    if ddi_key is None:
        xdt_name = f"{ant_key}"
    else:
        xdt_name = f"{ant_key}-{ddi_key}"
    output_xdt = xr.DataTree(dataset=output_xds.assign_coords(coords), name=xdt_name)
    print(output_xds["DELAYS"].values)
    return output_xdt


def _fit_data(coordinates, delays, locit_parms):
    """
    Execute the fitting using the desired engine, scipy or linear algebra
    Args:
        coordinates: the shape [4, : ] array with the ha, dec, elevation and time arrays
        delays: The delays to be fitted
        locit_parms: the locit input paramters

    Returns:
    fit: the fit results
    variance: the diagonal of the covariance matrix
    """
    try:
        ddi_id = locit_parms["this_ddi"]
    except KeyError:
        ddi_id = None
    label = create_dataset_label(locit_parms["this_ant"], ddi_id)

    fit_kterm = locit_parms["fit_kterm"]
    fit_rate = locit_parms["fit_delay_rate"]
    fit_engine = locit_parms["fit_engine"]

    if fit_engine == "linear algebra":
        try:
            fit, variance = _solve_linear_algebra(
                coordinates, delays, fit_kterm, fit_rate
            )
            return fit, variance, True
        except np.linalg.LinAlgError:
            logger.warning(
                f"Fitting failed for {label}, please try another fitting engine or DDI combination"
            )
            return np.nan, np.nan, False

    elif fit_engine == "scipy":
        try:
            fit, variance = _solve_scipy_optimize_curve_fit(
                coordinates, delays, fit_kterm, fit_rate, verbose=True
            )
            return fit, variance, True
        except TypeError:
            logger.warning(
                f"Fitting failed for {label}, please try another fitting engine or DDI combination"
            )
            return np.nan, np.nan, False

    else:
        msg = f'Unrecognized fitting engine: {locit_parms["fit_engine"]}'
        logger.error(msg)
        return np.nan, np.nan, False


def _compute_chi_squared(delays, fit, coordinates, fit_kterm, fit_rate):
    """
    Compute a model from fit results and computes the chi squared value of that model with respect to the data
    Args:
        delays: The observed delays
        fit: The fit results
        coordinates: ha, dec, elevation, time
        fit_kterm: K term fitted?
        fit_rate: delay rate fitted?

    Returns:
    The delay model and the chi squared value
    """
    model_function, _ = _define_fit_function(fit_kterm, fit_rate)
    model = model_function(coordinates, *fit)
    n_delays = len(delays)
    chi_squared = np.sum((model - delays) ** 2 / n_delays)
    return model, chi_squared


def _build_filtered_arrays(
    field_id, time, delays, locit_parms, antenna_info, source_dict
):
    """Build the coordinate arrays (ha, dec, elevation, time) for use in the fitting and filters data below the \
    elevation limit

    Args:
        field_id: Array with the observed field per delay
        time: Time array with the time of each delay
        delays: The delay array
        locit_parms: Locit main function parameters

    Returns:
    coordinates (ha, dec, ele, time), delays, local sidereal time all filtered by elevation limit and the \
    elevation_limit
    """
    elevation_limit = locit_parms["elevation_limit"] * convert_unit(
        "deg", "rad", "trigonometric"
    )
    geo_pos = antenna_info["geocentric_position"]
    ant_pos = EarthLocation.from_geocentric(geo_pos[0], geo_pos[1], geo_pos[2], "meter")
    astro_time = Time(time, format="mjd", scale="utc", location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian) / units.radian
    key = "precessed"

    n_samples = len(field_id)
    coordinates = np.ndarray([4, n_samples])
    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        coordinates[0:2, i_sample] = source_dict[field][key]
        coordinates[2, i_sample] = hadec_to_elevation(
            source_dict[field][key], antenna_info["latitude"]
        )
        coordinates[3, i_sample] = (
            time[i_sample] - time[0]
        )  # time is set to zero at the beginning of obs

    # convert to actual hour angle and wrap it to the [-pi, pi) interval
    coordinates[0, :] = lst.value - coordinates[0, :]
    coordinates[0, :] = np.where(
        coordinates[0, :] < 0, coordinates[0, :] + twopi, coordinates[0, :]
    )

    # Filter data below elevation limit
    selection = coordinates[2, :] > elevation_limit
    delays = delays[selection]
    coordinates = coordinates[:, selection]
    lst = lst[selection]
    nin = np.sum(selection)

    return coordinates, delays, lst, elevation_limit, nin


def _geometrical_coeffs(coordinates):
    """
    Compute the position related coefficients for the fitting, also the 1 corresponding to the fixed delay
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    the fixed delay coefficient (1), the x, y and z position delay coeffcients
    """
    ha, dec = coordinates[0:2]
    cosdec = np.cos(dec)
    xterm = np.cos(ha) * cosdec
    yterm = -np.sin(ha) * cosdec
    zterm = np.sin(dec)
    return [1.0, xterm, yterm, zterm]


def _kterm_coeff(coordinates):
    """Compute the k term (offset from antenna elevation axis) coefficient from elevation

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    The offset from antenna elevation axis delay coefficient
    """
    elevation = coordinates[2]
    return np.cos(elevation)


def _rate_coeff(coordinates):
    """Compute the delay rate coefficient (basically the time)

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    The delay rate coeeficient (time)
    """
    return coordinates[3]


def _solve_linear_algebra(coordinates, delays, fit_kterm, fit_rate):
    """

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        delays: The delays
        fit_kterm: fit elevation axis offset term
        fit_rate: fit delay rate term

    Returns:
    The fit results and the diagonal of the covariance matrix.
    """
    npar = 4 + fit_rate + fit_kterm

    system = np.zeros([npar, npar])
    vector = np.zeros([npar])
    n_samples = coordinates.shape[1]
    for i_sample in range(n_samples):
        coeffs = _system_coefficients(coordinates[:, i_sample], fit_kterm, fit_rate)
        for irow in range(npar):
            for icol in range(irow + 1):
                system[irow, icol] += coeffs[irow] * coeffs[icol]
            vector[irow] += delays[i_sample] * coeffs[irow]

    for irow in range(1, npar):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]

    fit, variance, _ = least_squares(system, vector)

    return fit, variance


def _system_coefficients(coordinates, fit_kterm, fit_rate):
    """Build coefficient list for linear algebra fit

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fit_kterm: fit elevation axis offset term
        fit_rate: Fit delay rate term

    Returns:

    """
    coeffs = _geometrical_coeffs(coordinates)
    if fit_kterm:
        coeffs.append(_kterm_coeff(coordinates))
    if fit_rate:
        coeffs.append(_rate_coeff(coordinates))
    return coeffs


def _define_fit_function(fit_kterm, fit_rate):
    """
    Define the fitting function based on the presence of the delay rate and elevation axis offset terms
    Args:
        fit_kterm: fit elevation axis offset?
        fit_rate: fit delay rate?

    Returns:
    The appropriate fitting function and the total number of parameters
    """
    npar = 4 + fit_rate + fit_kterm
    if fit_kterm and fit_rate:
        fit_function = _delay_model_kterm_rate
    elif fit_kterm and not fit_rate:
        fit_function = _delay_model_kterm_norate
    elif not fit_kterm and fit_rate:
        fit_function = _delay_model_nokterm_rate
    else:
        fit_function = _delay_model_nokterm_norate
    return fit_function, npar


def _solve_scipy_optimize_curve_fit(
    coordinates, delays, fit_kterm, fit_rate, verbose=False
):
    """
    Fit a delay model to the observed delays using scipy optimize curve_fit algorithm
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        delays: The observed delays
        fit_kterm: fit elevation axis offset term
        fit_rate: Fit delay rate term
        verbose: Display fitting messages

    Returns:
    The fit results and the diagonal of the covariance matrix
    """

    fit_function, npar = _define_fit_function(fit_kterm, fit_rate)

    # First guess is no errors in positions, no fixed delay and no delay rate
    p0 = np.zeros(npar)
    liminf = np.full(npar, -np.inf)
    limsup = np.full(npar, +np.inf)

    maxfevs = [100000, 1000000, 10000000]
    covar = None
    fit = None
    for maxfev in maxfevs:
        try:
            results = opt.curve_fit(
                fit_function,
                coordinates,
                delays,
                p0=p0,
                bounds=[liminf, limsup],
                maxfev=maxfev,
            )
            fit, covar = results[0:2]
        except RuntimeError:
            if verbose:
                logger.info("Increasing number of iterations")
                continue
            else:
                if verbose:
                    logger.info(
                        "Converged with less than {0:d} iterations".format(maxfev)
                    )
                break

    variance = np.diag(covar)
    return fit, variance


def _delay_model_nokterm_norate(coordinates, fixed_delay, xoff, yoff, zoff):
    """
    Delay model with no elevation axis offset or delay rate
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    return xterm + yterm + zterm + fixed_delay


def _delay_model_kterm_norate(coordinates, fixed_delay, xoff, yoff, zoff, koff):
    """
    Delay model with elevation axis offset and no delay rate
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        koff: Elevation axis offset delay

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm


def _delay_model_nokterm_rate(coordinates, fixed_delay, xoff, yoff, zoff, rate):
    """
    Delay model with delay rate and no elevation axis offset
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        rate: delay rate

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _rate_coeff(coordinates) * rate
    return xterm + yterm + zterm + fixed_delay + sterm


def _delay_model_kterm_rate(coordinates, fixed_delay, xoff, yoff, zoff, koff, rate):
    """
    Delay model with delay rate and elevation axis offset
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        koff: Elevation axis offset delay
        rate: delay rate

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _rate_coeff(coordinates) * rate
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm + sterm


def export_position_xds_to_table_row(
    row,
    attributes,
    del_fact,
    pha_fact,
    pos_fact,
    slo_fact,
    pos_unit,
    del_unit,
    kterm_present,
    rate_present,
):
    """
    Export the data from a single X array DataSet attributes to a table row (a list)
    Args:
        row: row onto which the data results are to be added
        attributes: The XDS attributes dictionary
        del_fact: Delay unit scaling factor
        pos_fact: Position unit scaling factor
        slo_fact: Delay rate unit scaling factor
        kterm_present: Is the elevation axis offset term present?
        rate_present: Is the delay rate term present?
        pha_fact: phase unit scaling factor
        pos_unit: Position unit
        del_unit: Delay unit

    Returns:
    The filled table row
    """

    delay_rms = np.sqrt(attributes["chi_squared"])
    mean_freq = np.nanmean(attributes["frequency"])
    phase_rms = twopi * mean_freq * delay_rms
    row.append(f"{delay_rms*del_fact:4.2e}")
    row.append(f"{phase_rms*pha_fact:5.1f}")

    sig_scale_pos = convert_unit("mm", pos_unit, "length")
    sig_scale_del = 1e-3 * convert_unit("nsec", del_unit, "time")

    row.append(
        fixed_format_error(
            attributes["fixed_delay_fit"],
            attributes["fixed_delay_error"],
            del_fact,
            sig_scale_del,
        )
    )
    position, poserr = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )

    for i_pos in range(3):
        row.append(
            fixed_format_error(position[i_pos], poserr[i_pos], pos_fact, sig_scale_pos)
        )
    if kterm_present:
        row.append(
            fixed_format_error(
                attributes["koff_fit"],
                attributes["koff_error"],
                pos_fact,
                sig_scale_pos,
            )
        )
    if rate_present:
        row.append(
            fixed_format_error(
                attributes["rate_fit"],
                attributes["rate_error"],
                slo_fact,
                sig_scale_del,
            )
        )
    return row


def export_position_xds_to_parminator(attributes, threshold, kterm_present):
    """
    Export a position xds attributes to a string ingestible by VLA's parminator
    :param attributes: xds attributes
    :param threshold: threshold of valid corrections in meters
    :param kterm_present: include K term in the parminator output
    :return: string Formated for parminator output
    """
    axes = ["X", "Y", "Z"]
    delays, _ = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )
    station = attributes["antenna_info"]["station"]

    outstr = ""
    for iaxis, delay in enumerate(delays):
        correction = delay * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,{axes[iaxis]},${correction: .4f}\n"

    if kterm_present:
        correction = attributes["koff_fit"] * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,K,${correction: .4f}\n"
    return outstr


def _compute_plot_borders(angle_fact, latitude, elevation_limit):
    """
    Compute plot limits and position of lines to be added to the plots
    Args:
        angle_fact: Angle scaling unit factor
        latitude: Antenna latitude
        elevation_limit: The elevation limit in the data set

    Returns:
    Elevation limits, elevation lines, declination limits, declination lines and hour angle limits
    """
    latitude *= angle_fact
    elevation_limit *= angle_fact
    right_angle = pi / 2 * angle_fact
    border = 0.05 * right_angle
    elelim = [-border, right_angle + border]
    border *= 2
    declim = [-border - right_angle + latitude, right_angle + border]
    border *= 2
    halim = [-border, 4 * right_angle + border]
    elelines = [0, elevation_limit]  # lines at zero and elevation limit
    declines = [latitude - right_angle, latitude + right_angle]
    return elelim, elelines, declim, declines, halim


def plot_antenna_position_corrections_worker(
    attributes_list, filename, telescope, ref_ant, parm_dict
):
    """
    Does the actual individual position correction plots
    Args:
        attributes_list: List of XDS attributes
        filename: Name of the PNG file to be created
        telescope: Telescope object used in observations
        ref_ant: Reference antenna in the data set
        parm_dict: Parameter dictionary of the caller's caller

    Returns:
    PNG file with the position corrections plot
    """
    tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
    length_unit = parm_dict["unit"]
    scaling = parm_dict["scaling"]
    len_fac = convert_unit("m", length_unit, "length")
    corr_fac = clight * scaling
    figure_size = parm_dict["figure_size"]
    box_size = parm_dict["box_size"]
    dpi = parm_dict["dpi"]
    display = parm_dict["display"]

    xlabel = f"East [{length_unit}]"
    ylabel = f"North [{length_unit}]"

    fig, axes = create_figure_and_axes(figure_size, [2, 2], default_figsize=[8, 8])
    xy_whole = axes[0, 0]
    xy_inner = axes[0, 1]
    z_whole = axes[1, 0]
    z_inner = axes[1, 1]

    for attributes in attributes_list:
        antenna = attributes["antenna_info"]
        ew_off, ns_off, _, _ = compute_antenna_relative_off(
            antenna, tel_lon, tel_lat, tel_rad, len_fac
        )
        corrections, _ = rotate_to_gmt(
            np.copy(attributes["position_fit"]),
            attributes["position_error"],
            antenna["longitude"],
        )
        corrections = np.array(corrections) * corr_fac
        text = "  " + antenna["name"]
        if antenna["name"] == ref_ant:
            text += "*"
        plot_one_antenna_position(
            xy_whole, xy_inner, ew_off, ns_off, text, box_size, marker="+"
        )
        add_antenna_position_corrections_to_plot(
            xy_whole, xy_inner, ew_off, ns_off, corrections[0], corrections[1], box_size
        )
        plot_one_antenna_position(
            z_whole, z_inner, ew_off, ns_off, text, box_size, marker="+"
        )
        add_antenna_position_corrections_to_plot(
            z_whole, z_inner, ew_off, ns_off, 0, corrections[2], box_size
        )

    plot_boxes_limits_and_labels(
        xy_whole,
        xy_inner,
        xlabel,
        ylabel,
        box_size,
        "X & Y, outer array",
        "X & Y, inner array",
    )
    plot_boxes_limits_and_labels(
        z_whole, z_inner, xlabel, ylabel, box_size, "Z, outer array", "Z, inner array"
    )
    close_figure(fig, "Position corrections", filename, dpi, display)


def add_antenna_position_corrections_to_plot(
    outerax, innerax, xpos, ypos, xcorr, ycorr, box_size, color="red", linewidth=0.5
):
    """
    Plot an antenna position corrections as a vector from the antenna position
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        xcorr: X axis correction (horizontal on plot)
        ycorr: Y axis correction (vectical on plot)
        box_size: inner array box size
        color: vector color
        linewidth: vector line width
    """
    half_box = box_size / 2
    head_size = np.sqrt(xcorr**2 + ycorr**2) / 4
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )
    else:
        outerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )
        innerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )
