import argparse
import glob
import time
import numpy as np

import casatools

from astrohack import extract_locit, locit, open_locit, open_position
from astrohack.utils.algorithms import rotate_to_gmt, data_statistics
from astrohack.utils.constants import clight, pol_str
from astrohack.utils.conversion import convert_unit
from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    run_casatask,
    proceed_check,
    list_input_tooltip,
    run_astrohack_function,
    parse_list_or_all,
    base_name_determination,
    asdm_test_and_import,
    add_basic_info_and_parameters_to_report,
)
from astrohack.utils.text import (
    format_duration,
    create_html_file_from_body,
    lnbr,
    create_single_html_image_with_header,
    add_preformatted_text_file_to_html,
    add_heading_to_html,
    make_collapsible_block,
)
from astrohack.visualization import create_figure_and_axes, close_figure
from astrohack.visualization.plot_tools import filter_duplicates_in_legend


def parse():
    desc = "CASA baseline pipeline"

    parser = argparse.ArgumentParser(
        description=f"{desc}", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("filename", type=str, help="Path to the input MS/ASDM file")

    parser.add_argument("refant", type=str, help="Reference antenna for calibration")

    parser.add_argument(
        "-r",
        "--root-name",
        type=str,
        default=None,
        help="Root name for the calibration tables, default is filename without extension",
    )

    parser.add_argument(
        "-f",
        "--fringefit_source",
        default="0319+415",
        help="Fringe fit source, default is 0319+415",
    )
    parser.add_argument(
        "--scans_to_flag",
        default=None,
        type=str,
        help="Comma separated list of scans to flag, default is None",
    )

    parser.add_argument(
        "-i",
        "--intent",
        default="CALIBRATE_POINTING#ON_SOURCE",
        type=str,
        help="Intent for pointing observations.",
    )

    parser.add_argument(
        "-s",
        "--spw",
        type=str,
        default="all",
        help=f"Select SPWs for locit processing, {list_input_tooltip('0,1,2')}, default is %(default)s",
    )

    parser.add_argument(
        "-a",
        "--antenna",
        default="all",
        help="Select antennas for which to produce antenna position corrections, "
        f"{list_input_tooltip('ea01,ea02')}, default is %(default)s"
        "",
    )

    parser.add_argument(
        "-e",
        "--elevation-limit",
        type=float,
        default=10.0,
        help="Lowest elevation of data for consideration in degrees, default is %(default).1f",
    )

    parser.add_argument(
        "-p",
        "--polarization",
        type=str,
        choices=["both", "L", "R"],
        default="both",
        help="Which polarization hands to be used for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-c",
        "--combination",
        type=str,
        choices=["simple", "difference"],
        default="simple",
        help="How to combine different spws for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-k",
        "--fit_kterm",
        action="store_true",
        default=False,
        help="Fit antennas K term (i.e. Offset between azimuth and elevation axes)",
    )

    parser.add_argument(
        "-l",
        "--delay_limits",
        type=str,
        default="-0.1,0.1",
        help='Delay limits for delay plots, values must be given between quotes("), default is "%(default)s"',
    )

    parser.add_argument(
        "-d",
        "--dpi",
        type=int,
        default=300,
        help="DPI for png figures (default: %(default)s)",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing files (MSes, caltables, locit files, plots)",
    )

    parser.add_argument(
        "--starting-stage",
        type=str,
        default="calibration",
        choices=["calibration", "locit", "exports", "report"],
        help="Starting stage in which to start processing (default: %(default)s).",
    )

    parser.add_argument(
        "--reimport-asdm",
        action="store_true",
        default=False,
        help="Forcefully re-import the asdm file is the ms already exists (default: %(default)s)",
    )

    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Assume yes on proceed."
    )

    return vars(parser.parse_args())


def param_init(param_dict: dict, msger: MessageBoard):
    base_name = base_name_determination(param_dict)
    param_dict = asdm_test_and_import(param_dict, base_name, msger)

    param_dict["pointing_only_ms"] = f"{base_name}.pnt.ms"
    param_dict["freq_averaged_ms"] = f"{base_name}.avg.ms"
    param_dict["fringefit_caltable"] = f"{base_name}.sbd"
    param_dict["phase_caltable"] = f"{base_name}.pha.gcal"
    param_dict["antpos_caltable"] = f"{base_name}.antpos"
    param_dict["locit_name"] = f"{base_name}.locit.zarr"
    param_dict["position_name"] = f"{base_name}.position.zarr"
    param_dict["exports_name"] = f"{base_name}.exports"
    param_dict["report_name"] = f"{base_name}-report.html"

    param_dict["antenna"] = parse_list_or_all(param_dict, "antenna")
    param_dict["spw"] = parse_list_or_all(param_dict, "spw", list_type=int)
    param_dict["delay_limits"] = parse_list_or_all(
        param_dict, "delay_limits", list_type=float, max_size=2
    )

    if param_dict["scans_to_flag"] is None:
        param_dict["scans_to_flag"] = []
    else:
        param_dict["scans_to_flag"] = param_dict["scans_to_flag"].split(",")

    # Ms data fetching and some consistency checks
    pnt_intent = "CALIBRATE_POINTING#ON_SOURCE"
    msmd = casatools.msmetadata()
    msmd.open(param_dict["msname"])
    ant_names = msmd.antennanames()
    field_names = msmd.fieldnames()
    spw_list = msmd.spwsforintent(pnt_intent)
    nchan = np.unique([msmd.nchan(i_spw) for i_spw in spw_list])
    msmd.done()

    param_dict["n_chan"] = nchan[0]

    error_msgs = []
    if param_dict["refant"] not in ant_names:
        error_msgs.append(f"Chosen refant ({param_dict['refant']}) not present in ms.")
    if param_dict["fringefit_source"] not in field_names:
        error_msgs.append(
            f"Chosen fringefit source ({param_dict['fringefit_source']}) not present in ms."
        )
    if nchan.size != 1:
        error_msgs.append(
            "Spectral windows are not consistent with each other, is this really a pointing ms?"
        )
    if len(error_msgs) > 0:
        raise RuntimeError("\n".join(error_msgs))

    initialization_check(param_dict, "Baseline determination parameters")
    return param_dict


def run_casa_pre_locit_steps(param_dict: dict, msger: MessageBoard):
    run_casatask(
        "split",
        {
            "vis": param_dict["msname"],
            "outputvis": param_dict["pointing_only_ms"],
            "intent": param_dict["intent"],
            "datacolumn": "data",
        },
        msger,
        intended_output=param_dict["pointing_only_ms"],
        overwrite=param_dict["overwrite"],
    )

    if len(param_dict["scans_to_flag"]) > 0:
        run_casatask(
            "flagdata",
            {
                "vis": param_dict["pointing_only_ms"],
                "mode": "manual",
                "scan": ",".join(param_dict["scans_to_flag"]),
                "action": "apply",
                "display": "report",
                "flagbackup": False,
            },
            msger,
        )
        run_casatask(
            "flagmanager",
            {
                "vis": param_dict["pointing_only_ms"],
                "mode": "save",
                "versionname": "baseflags",
            },
            msger,
        )

    fringefit_was_run = run_casatask(
        "fringefit",
        {
            "vis": param_dict["pointing_only_ms"],
            "caltable": param_dict["fringefit_caltable"],
            "field": param_dict["fringefit_source"],
            "solint": "inf",
            "refant": param_dict["refant"],
            "minsnr": 3.0,
            "zerorates": True,
            "globalsolve": True,
            "niter": 100,
        },
        msger,
        intended_output=param_dict["fringefit_caltable"],
        overwrite=param_dict["overwrite"],
    )
    if fringefit_was_run:
        run_casatask(
            "applycal",
            {
                "vis": param_dict["pointing_only_ms"],
                "gaintable": [param_dict["fringefit_caltable"]],
                "interp": ["nearest"],
                "parang": False,
            },
            msger,
        )

    # Now we create a new dataset that is colapsed on the channel axis
    # within each spw, also create a flagversion to store current flag
    # state on the averaged MS
    freq_average_was_run = run_casatask(
        "split",
        {
            "vis": param_dict["pointing_only_ms"],
            "outputvis": param_dict["freq_averaged_ms"],
            "datacolumn": "corrected",
            "keepflags": False,  #
            "width": param_dict["n_chan"],
        },
        msger,
        intended_output=param_dict["freq_averaged_ms"],
        overwrite=param_dict["overwrite"],
    )
    if freq_average_was_run:
        run_casatask(
            "flagmanager",
            {
                "vis": param_dict["freq_averaged_ms"],
                "mode": "save",
                "versionname": "original",
            },
            msger,
        )

    gaincal_was_run = run_casatask(
        "gaincal",
        {
            "vis": param_dict["freq_averaged_ms"],
            "caltable": param_dict["phase_caltable"],
            "solint": "10min",
            "refant": param_dict["refant"],
            "refantmode": "flex",  # Maybe we should use strict for this application?
            "minblperant": 3,
            "minsnr": 3.0,
            "gaintype": "G",  # G is for gain
            "calmode": "p",  # p is for phase
            "solmode": "L1",  # -> least squares
        },
        msger,
        intended_output=param_dict["phase_caltable"],
        overwrite=param_dict["overwrite"],
    )
    if gaincal_was_run:
        run_casatask(
            "applycal",
            {
                "vis": param_dict["freq_averaged_ms"],
                "gaintable": [param_dict["phase_caltable"]],
                "interp": ["nearest"],
                "parang": False,
            },
            msger,
        )

    if not param_dict["assume_yes"]:
        run_casatask(
            "plotms",
            {
                "vis": param_dict["freq_averaged_ms"],
                "xaxis": "time",
                "yaxis": "phase",
                "ydatacolumn": "corrected",
                "field": "*",
                "avgtime": "10",
                "correlation": "RR,LL",
                "coloraxis": "spw",
                "antenna": param_dict["refant"],
                "iteraxis": "baseline",
            },
            msger,
        )
        proceed_check(param_dict, "Are phases clustered around 0 in plotMS?")

    return


def run_astrohack_locit(param_dict: dict, msger: MessageBoard):
    astrohack_param_dict = {
        "cal_table": param_dict["phase_caltable"],
        "locit_name": param_dict["locit_name"],
        "position_name": param_dict["position_name"],
        "ant": param_dict["antenna"],
        "ddi": param_dict["spw"],
        "overwrite": param_dict["overwrite"],
        "fit_kterm": param_dict["fit_kterm"],
        "fit_delay_rate": True,
        "elevation_limit": param_dict["elevation_limit"],
        "polarization": param_dict["polarization"],
        "combine_ddis": param_dict["combination"],
        "parallel": False,
    }
    locit_functions = [extract_locit, locit]
    for function in locit_functions:
        status, exec_exception = run_astrohack_function(
            astrohack_param_dict, function, msger
        )
        if not status:
            raise RuntimeError(
                f"{function.__name__} failed see above for details."
            ) from exec_exception


def run_astrohack_exports(param_dict: dict, msger: MessageBoard):
    astrohack_param_dict = {
        "destination": param_dict["exports_name"],
        "ant": param_dict["antenna"],
        "filename": f"{param_dict['exports_name']}/parminator.par",
        "parallel": False,
        "dpi": param_dict["dpi"],
        "delay_limits": param_dict["delay_limits"],
    }
    locit_mds = open_locit(param_dict["locit_name"])
    position_mds = open_position(param_dict["position_name"])
    plotting_methods = [
        locit_mds.plot_source_positions,
        locit_mds.plot_array_configuration,
        position_mds.plot_delays,
        position_mds.plot_position_corrections,
        position_mds.export_locit_fit_results,
        position_mds.export_results_to_parminator,
    ]
    for plot_method in plotting_methods:
        status, exec_exception = run_astrohack_function(
            astrohack_param_dict, plot_method, msger
        )
        if not status:
            raise RuntimeError(
                f"{plot_method.__name__} failed see above for details."
            ) from exec_exception

    return


def make_single_antenna_phase_over_time(
    exports_name,
    refant_name,
    ant_name,
    corr_axis,
    scan_list,
    plt_times,
    raw_phases,
    corr_phases,
    dpi,
):
    import matplotlib.ticker as ticker

    def forward(x_val):
        return np.interp(x_val, plt_times, scan_list)

    def inverse(x_val):
        return np.interp(x_val, scan_list, plt_times)

    corr_styles = ["+", "x"]
    fig, ax = create_figure_and_axes(None, [1, 1])
    for i_corr, corr in enumerate(corr_axis):
        legend = f"{corr}"
        ax.plot(
            plt_times,
            raw_phases[:, :, i_corr],
            f"r{corr_styles[i_corr]}",
            label=f"DATA, {legend}",
        )
        ax.plot(
            plt_times,
            corr_phases[:, :, i_corr],
            f"b{corr_styles[i_corr]}",
            label=f"CORRECTED, {legend}",
        )
    raw_rms = data_statistics(raw_phases)["rms"]
    corr_rms = data_statistics(corr_phases)["rms"]
    ax.legend()
    ax.set_xlabel("Time since observation start [hour]")
    ax.set_ylabel("Phase [deg]")
    ax.set_ylim([-180, 180])
    secax = ax.secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xlabel("Scan number")
    secax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    if corr_rms < 0.3 * raw_rms:
        status = "Greatly improved!"
    elif corr_rms < 0.8 * raw_rms:
        status = "Improved!"
    else:
        status = "Not improved..."
    ax.set_title(
        f"data RMS: {raw_rms:.2f} [deg], corrected RMS: {corr_rms:.2f} [deg] ({status})"
    )
    filter_duplicates_in_legend(ax)
    close_figure(
        fig,
        f"Phases before and after antenna position corrections for {ant_name}&{refant_name}",
        f"{exports_name}/phases-antpos-{ant_name}.png",
        dpi,
        False,
        True,
    )
    return


def make_all_antenna_phases_plot(
    param_dict: dict, ant_names: list, msger: MessageBoard
):
    from casacoretables import tables

    start = time.time()
    msger.one_liner("Preparing phase over time plots...")

    rad2deg = convert_unit("rad", "deg", "trigonometric")
    sec2hr = convert_unit("sec", "hour", "time")

    avg_ms = param_dict["freq_averaged_ms"]
    ant_table = tables.table(avg_ms + "/ANTENNA")
    ms_ant_names = ant_table.getcol("NAME")
    ant_table.close()

    pol_table = tables.table(avg_ms + "/POLARIZATION")
    corr_axis = pol_str[pol_table.getcol("CORR_TYPE")[0, :]]
    pol_table.close()
    corr_select = np.logical_or(corr_axis == "RR", corr_axis == "LL")
    corr_axis = corr_axis[corr_select]

    ms_table = tables.table(avg_ms)
    vis_time = ms_table.getcol("TIME")
    vis_raw = ms_table.getcol("DATA")[:, 0, corr_select]
    vis_corr = ms_table.getcol("CORRECTED_DATA")[:, 0, corr_select]
    ant_1 = ms_table.getcol("ANTENNA1")
    ant_2 = ms_table.getcol("ANTENNA2")
    scans = ms_table.getcol("SCAN_NUMBER")
    weights = ms_table.getcol("WEIGHT")[:, corr_select]
    flags = ms_table.getcol("FLAG")[:, 0, corr_select]
    spws = ms_table.getcol("DATA_DESC_ID")
    ms_table.close()

    spw_list = np.unique(spws)
    scan_list = np.unique(scans)
    n_spws = spw_list.shape[0]
    n_scans = scan_list.shape[0]
    n_corrs = corr_axis.shape[0]

    first_time = np.min(np.unique(vis_time))

    refant_id = ms_ant_names.index(param_dict["refant"])
    for ant_name in ant_names:
        if ant_name == param_dict["refant"]:
            continue
        ant_id = ms_ant_names.index(ant_name)
        ant_select = np.logical_or(
            np.logical_and(ant_1 == ant_id, ant_2 == refant_id),
            np.logical_and(ant_2 == ant_id, ant_1 == refant_id),
        )
        ant_sel_scans = scans[ant_select]
        ant_sel_flags = flags[ant_select]
        ant_sel_time = vis_time[ant_select]

        ant_sel_raw = np.ma.masked_array(vis_raw[ant_select], ant_sel_flags)
        ant_sel_corr = np.ma.masked_array(vis_corr[ant_select], ant_sel_flags)
        ant_sel_weights = weights[ant_select]
        ant_sel_spws = spws[ant_select]

        phase_shape = (n_scans, n_spws, n_corrs)
        plt_times = np.full_like(scan_list, np.nan, dtype=float)
        raw_phases = np.full(phase_shape, np.nan, dtype=float)
        corr_phases = np.full(phase_shape, np.nan, dtype=float)
        for i_scan, scan in enumerate(scan_list):
            scan_select = ant_sel_scans == scan
            plt_times[i_scan] = (
                np.average(ant_sel_time[scan_select]) - first_time
            ) * sec2hr
            scan_sel_weights = ant_sel_weights[scan_select]
            scan_sel_corr = ant_sel_corr[scan_select]
            scan_sel_raw = ant_sel_raw[scan_select]
            scan_sel_spws = ant_sel_spws[scan_select]
            for i_spw, spw in enumerate(spw_list):
                spw_select = scan_sel_spws == spw
                avg_raw = np.ma.average(
                    scan_sel_raw[spw_select],
                    weights=scan_sel_weights[spw_select],
                    axis=0,
                )
                raw_phases[i_scan, i_spw, :] = np.angle(avg_raw) * rad2deg
                avg_corr = np.ma.average(
                    scan_sel_corr[spw_select],
                    weights=scan_sel_weights[spw_select],
                    axis=0,
                )
                corr_phases[i_scan, i_spw, :] = np.angle(avg_corr) * rad2deg

        make_single_antenna_phase_over_time(
            param_dict["exports_name"],
            param_dict["refant"],
            ant_name,
            corr_axis,
            scan_list,
            plt_times,
            raw_phases,
            corr_phases,
            param_dict["dpi"],
        )

    stop = time.time()
    msger.one_liner(f"Plotting finished in {format_duration(stop - start)}")
    return


def run_post_locit_plots(param_dict: dict, msger: MessageBoard):
    pos_corrections = []
    ant_names = []
    position_mds = open_position(param_dict["position_name"])
    for ant_key, ant_xdt in position_mds.items():
        attributes = ant_xdt.attrs
        geo_delays, _ = rotate_to_gmt(
            np.copy(attributes["position_fit"]),
            attributes["position_error"],
            attributes["antenna_info"]["longitude"],
        )
        pos_corrections.extend(clight * geo_delays)
        ant_names.append(attributes["antenna_info"]["name"])

    gencal_was_run = run_casatask(
        "gencal",
        {
            "vis": param_dict["freq_averaged_ms"],
            "caltable": param_dict["antpos_caltable"],
            "caltype": "antpos",
            "antenna": ",".join(ant_names),
            "parameter": pos_corrections,
        },
        msger,
        intended_output=param_dict["antpos_caltable"],
        overwrite=param_dict["overwrite"],
    )
    if gencal_was_run:
        run_casatask(
            "applycal",
            {
                "vis": param_dict["freq_averaged_ms"],
                "gaintable": [param_dict["antpos_caltable"]],
                "interp": ["nearest"],
                "parang": False,
            },
            msger,
        )

    make_all_antenna_phases_plot(param_dict, ant_names, msger)
    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    images_to_include = {
        "locit_source_table_fk5.png": "Source positions over the sky",
        "locit_array_configuration.png": "VLA configuration during observation",
        "position_corrections_combined_simple.png": "Graphical representation of antenna position corrections",
    }

    report_title = f"Baseline Report for {param_dict['filename']}"
    html_body = add_heading_to_html(report_title, 1)

    html_body += add_basic_info_and_parameters_to_report(param_dict)

    for image_file, image_desc in images_to_include.items():
        image_path = f"{exports_name}/{image_file}"
        html_body += (
            f"{create_single_html_image_with_header(image_path, image_desc)}{lnbr}"
        )

    html_body += add_preformatted_text_file_to_html(
        f"{exports_name}/position_combined_simple_fit_results.txt",
        "Measured antenna position corrections",
    )

    html_body += add_preformatted_text_file_to_html(
        f"{exports_name}/parminator.par",
        "Proposed parminator corrections",
    )

    delay_plot_files = glob.glob(f"{exports_name}/position_delays_ant_*.png")
    antenna_name_list = np.sort(
        [dpf.split("/")[1].split("_")[3] for dpf in delay_plot_files]
    )

    for ant_name in antenna_name_list:
        if ant_name != param_dict["refant"]:
            delay_plot_file = f"{exports_name}/position_delays_ant_{ant_name}_combined_{param_dict["combination"]}.png"
            ant_html = create_single_html_image_with_header(
                delay_plot_file,
                f"Measured and fitted delays",
                heading_level=3,
            )
            ant_html += create_single_html_image_with_header(
                f"{exports_name}/phases-antpos-{ant_name}.png",
                f"Phase before and after correction",
                heading_level=3,
            )
            html_body += make_collapsible_block(
                ant_html,
                add_heading_to_html(f"Results for {ant_name}", 2),
                f"ant_{ant_name}",
            )
    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner("Report finished in {:.2f} seconds".format(stop - start))


def main():
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.heading("Welcome to the AstroHACK baseline pipeline for the VLA")

    param_dict = param_init(parse(), msger)
    processing_stage = param_dict["starting_stage"]

    if processing_stage == "calibration":
        run_casa_pre_locit_steps(param_dict, msger)
        processing_stage = "locit"

    if processing_stage == "locit":
        run_astrohack_locit(param_dict, msger)
        processing_stage = "exports"

    if processing_stage == "exports":
        run_astrohack_exports(param_dict, msger)
        run_post_locit_plots(param_dict, msger)
        processing_stage = "report"

    if processing_stage == "report":
        prepare_html_report(param_dict, msger)

    pipeline_end = time.time()
    msger.heading(
        f"Baseline processing finished in {format_duration(pipeline_end-pipeline_start)}, "
        + f"locit results (including parminator file) saved at: {param_dict['exports_name']}."
        + f" Checkout the HTML report at: {param_dict['report_name']}."
    )
    return
