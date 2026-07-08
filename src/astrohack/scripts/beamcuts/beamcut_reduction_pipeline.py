import argparse
import time
import casatools

import numpy as np
from toolviper.dask.client import local_client
from astrohack import (
    extract_pointing,
    extract_holog,
    beamcut,
    open_pointing,
    open_beamcut,
)
from astrohack.utils.pipeline_support import (
    initialization_check,
    MessageBoard,
    list_input_tooltip,
    base_name_determination,
    asdm_test_and_import,
    parse_list_or_all,
    run_casatask,
    run_astrohack_function,
    add_basic_info_and_parameters_to_report,
)
from astrohack.utils.text import (
    format_duration,
    create_html_file_from_body,
    add_heading_to_html,
    create_single_html_image_with_header,
    make_collapsible_block,
    add_preformatted_text_file_to_html,
)


def parse():
    parser = argparse.ArgumentParser(description="Beam cut reduction pipeline")

    parser.add_argument(
        "filename", type=str, help="Path to the input dataset to process."
    )

    parser.add_argument("refant", type=str, help="Reference antenna for calibration")

    parser.add_argument(
        "-r",
        "--root-name",
        type=str,
        default=None,
        help="Root name for the products of the pipeline, default"
        " is ms_name without extension",
    )

    parser.add_argument(
        "-q",
        "--quack-nchan",
        default=4,
        type=int,
        help="Number of channels to quack at the edge of the spectral window (default is %(default)s)",
    )

    parser.add_argument(
        "-f",
        "--beamcut-field",
        default=None,
        type=str,
        help="Field Id or name of the beam cut data (default is to determine it from data)",
    )

    parser.add_argument(
        "-s",
        "--spectral-window",
        type=str,
        default="all",
        help=f"Select SPWs for which to produce beam cuts, {list_input_tooltip('0,1,2')}, default is %(default)s",
    )

    parser.add_argument(
        "-a",
        "--antenna",
        type=str,
        default="all",
        help="Select antennas for which to produce beam cuts, "
        f"{list_input_tooltip('ea01,ea02')}, default is %(default)s",
    )

    parser.add_argument(
        "-n",
        "--ncores",
        type=int,
        default=4,
        help="Number of cores to use, default is %(default)d",
    )

    parser.add_argument(
        "-m",
        "--memory-per-core",
        type=str,
        default="10GB",
        help="Memory per core to use, default is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if found",
    )

    parser.add_argument(
        "-d",
        "--data-column",
        type=str,
        default="CORRECTED_DATA",
        help="Data column to be extracted from MS, default is %(default)s",
    )

    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Assume yes on proceed."
    )

    # Example of parameter with choice
    parser.add_argument(
        "--starting-stage",
        type=str,
        default="calibration",
        choices=[
            "calibration",
            "extract_pointing",
            "extract_holog",
            "beamcut",
            "exports",
            "report",
        ],
        help="Starting stage in which to start processing (default: %(default)s).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots Per Inch for plotting, default is %(default)d",
    )

    parser.add_argument(
        "--plot-pointing",
        action="store_true",
        help="Plot antenna pointing, default is %(default)s",
    )

    parser.add_argument(
        "--exclude-bad-antennas",
        default=None,
        type=str,
        help=f"Exclude antennas with bad data, {list_input_tooltip('ea18,ea01')}, default is %(default)s.",
    )

    return vars(parser.parse_args())


def fetch_ms_metadata(param_dict: dict):
    # Fetch metadata from ms
    msmd = casatools.msmetadata()
    msmd.open(param_dict["msname"])
    cal_scans = msmd.scansforintent("*PHASE*")
    beamcut_scans = msmd.scansforintent("*MAP*ON_SOURCE")
    spw_list = msmd.spwsforintent("*MAP*")
    beamcut_fields = np.unique(msmd.fieldsforscans(beamcut_scans))
    nchan = np.unique([msmd.nchan(i_spw) for i_spw in spw_list])
    all_fields = msmd.fieldnames()
    msmd.done()

    if param_dict["beamcut_field"] is None:
        if beamcut_fields.size > 1:
            raise RuntimeError("More than 1 beam cut field, try splitting the ms")
        param_dict["beamcut_field"] = beamcut_fields[0]
    else:
        try:
            field_id = int(param_dict["beamcut_field"])
            if field_id > all_fields.size - 1 or field_id < 0:
                raise RuntimeError("Specified beam cut field ID is out of range")
        except ValueError:
            if param_dict["beamcut_field"] not in all_fields:
                raise RuntimeError(
                    f"{param_dict['beamcut_field']} not present in the ms"
                )

    if nchan.size > 1:
        raise RuntimeError(
            "Spectral windows have different nchans, don't know how to proceed automatically"
        )

    # Convert to comma-separated string
    param_dict["calibration_scans"] = ",".join(map(str, cal_scans))
    param_dict["beamcut_scans"] = ",".join(map(str, beamcut_scans))

    fchan = param_dict["quack_nchan"]
    lchan = nchan[0] - param_dict["quack_nchan"]
    minspw = f"{round(np.min(spw_list)):d}"
    maxspw = f"{round(np.max(spw_list)):d}"
    spwrange = f"{minspw}~{maxspw}"
    param_dict["quacked_spw_selection"] = f"{spwrange}:{fchan}~{lchan}"
    return param_dict


def param_init(param_dict: dict, msger: MessageBoard):
    extensions = {
        "delay_cal": ".dcal",
        "bandpass_cal": ".bcal",
        "gain_cal": ".gcal",
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "beamcut": ".beamcut.zarr",
        "exports": ".exports",
        "report": "-report.html",
    }

    base_name = base_name_determination(param_dict)
    param_dict = asdm_test_and_import(param_dict, base_name, msger)

    for identifier, extension in extensions.items():
        param_dict[f"{identifier}_name"] = base_name + extension

    param_dict = fetch_ms_metadata(param_dict)

    param_dict["antenna"] = parse_list_or_all(param_dict["antenna"])
    param_dict["spectral_window"] = parse_list_or_all(param_dict["spectral_window"])

    if param_dict["exclude_bad_antennas"] is not None:
        param_dict["exclude_bad_antennas"] = parse_list_or_all(
            param_dict["exclude_bad_antennas"]
        )
    param_dict["parallel"] = param_dict["ncores"] >= 2
    initialization_check(param_dict, "Beam cut reduction parameters")
    return param_dict


def run_casa_calibration(param_dict, msger):
    gaintable = []
    delay_ok = run_casatask(
        "gaincal",
        {
            "vis": param_dict["msname"],
            "caltable": param_dict["delay_cal_name"],
            "refant": param_dict["refant"],
            "solint": "inf",
            "spw": param_dict["quacked_spw_selection"],
            "scan": param_dict["calibration_scans"],
            "gaintype": "K",
        },
        msger,
        intended_output=param_dict["delay_cal_name"],
        overwrite=param_dict["overwrite"],
    )

    if delay_ok:
        gaintable.append(param_dict["delay_cal_name"])
        bandpass_ok = run_casatask(
            "bandpass",
            {
                "vis": param_dict["msname"],
                "caltable": param_dict["bandpass_cal_name"],
                "refant": param_dict["refant"],
                "solint": "10s",
                "spw": param_dict["quacked_spw_selection"],
                "scan": param_dict["calibration_scans"],
                "solnorm": True,
                "gaintable": gaintable,
            },
            msger,
            intended_output=param_dict["bandpass_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        bandpass_ok = False

    if bandpass_ok:
        gaintable.append(param_dict["bandpass_cal_name"])
        gaincal_ok = run_casatask(
            "gaincal",
            {
                "vis": param_dict["msname"],
                "caltable": param_dict["gain_cal_name"],
                "refant": param_dict["refant"],
                "calmode": "ap",
                "solint": "inf",
                "spw": param_dict["quacked_spw_selection"],
                "minsnr": 2,
                "minblperant": 2,
                "scan": param_dict["calibration_scans"],
                "gaintable": gaintable,
            },
            msger,
            intended_output=param_dict["gain_cal_name"],
            overwrite=param_dict["overwrite"],
        )
    else:
        gaincal_ok = False

    if gaincal_ok:
        gaintable.append(param_dict["gain_cal_name"])
        run_casatask(
            "applycal",
            {
                "vis": param_dict["msname"],
                "field": f"{param_dict["beamcut_field"]}",
                "spw": param_dict["quacked_spw_selection"],
                "applymode": "calonly",
                "gaintable": gaintable,
            },
            msger,
        )

    return


def run_astrohack_reduction(param_dict, msger):
    # Astrohack convenience changes
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spectral_window"]
    param_dict["exclude_antennas"] = param_dict["exclude_bad_antennas"]

    param_dict["ms_name"] = param_dict["msname"]

    status = True
    exec_exception = None
    exec_list = [
        ["extract_holog", extract_pointing],
        ["beamcut", extract_holog],
        ["exports", beamcut],
    ]
    for next_stage, function in exec_list:
        if status:
            status, exec_exception = run_astrohack_function(
                param_dict,
                function,
                msger,
            )
            if status:
                param_dict["processing_stage"] = next_stage

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def run_astrohack_exports(param_dict, msger):
    param_dict["destination"] = param_dict["exports_name"]
    pnt_mds = open_pointing(param_dict["point_name"])
    if pnt_mds is None:
        raise RuntimeError(f"{param_dict['point_name']} not found")
    bmc_mds = open_beamcut(param_dict["beamcut_name"])
    if bmc_mds is None:
        raise RuntimeError(f"{param_dict['beamcut_name']} not found")

    plotting_methods = [
        pnt_mds.plot_array_configuration,
        bmc_mds.plot_in_amplitude,
        bmc_mds.plot_in_phase,
        bmc_mds.plot_in_db,
        bmc_mds.plot_lm_offsets,
        bmc_mds.export_report,
    ]
    if param_dict["plot_pointing"]:
        param_dict["plot_antennas_separately"] = True
        plotting_methods.append(pnt_mds.plot_pointing_in_time)

    for plot_method in plotting_methods:
        status, exec_exception = run_astrohack_function(param_dict, plot_method, msger)
        if not status:
            raise RuntimeError(
                f"{plot_method.__name__} failed see above for details."
            ) from exec_exception

    return


def prepare_html_report(param_dict, msger):
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    report_title = f"Beamcut report for {param_dict['filename']}"

    html_body = add_heading_to_html(report_title, 1)
    html_body += add_basic_info_and_parameters_to_report(param_dict)
    html_body += create_single_html_image_with_header(
        f"{exports_name}/point_array_configuration.png",
        "Array configuration during observation",
        heading_level=2,
    )

    bmc_mds = open_beamcut(param_dict["beamcut_name"])
    if bmc_mds is None:
        raise RuntimeError(f"{param_dict['beamcut_name']} not found")
    antenna_list = [ant_key.split("_")[-1] for ant_key in bmc_mds.keys()]
    ddi_list = [
        ddi_key.split("_")[-1] for ddi_key in bmc_mds[f"ant_{antenna_list[0]}"].keys()
    ]

    for ant_name in antenna_list:
        ant_html = ""
        if param_dict["plot_pointing"]:
            ant_html += create_single_html_image_with_header(
                f"{exports_name}/point_directional_cosines_ant_{ant_name}.png",
                "Pointing over time:",
                heading_level=3,
            )
            ant_html += create_single_html_image_with_header(
                f"{exports_name}/beamcut_lm_offsets_ant_{ant_name}_ddi_{ddi_list[0]}.png",
                "Pointing over sky:",
                heading_level=3,
            )
        for ddi_name in ddi_list:
            spw_html = create_single_html_image_with_header(
                f"{exports_name}/beamcut_db_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in dB",
                heading_level=4,
            )
            spw_html += create_single_html_image_with_header(
                f"{exports_name}/beamcut_amplitude_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in amplitude",
                heading_level=4,
            )
            spw_html += create_single_html_image_with_header(
                f"{exports_name}/beamcut_phase_ant_{ant_name}_ddi_{ddi_name}.png",
                "Beam cut in phase",
                heading_level=4,
            )
            spw_html += add_preformatted_text_file_to_html(
                f"{exports_name}/beamcut_report_ant_{ant_name}_ddi_{ddi_name}.txt",
                "Beam cut fit report",
                heading_level=4,
            )
            ant_html += make_collapsible_block(
                spw_html,
                add_heading_to_html(f"\t{ant_name} spectral window {ddi_name}:", 3),
            )
        # collapsible wrapping here
        html_body += make_collapsible_block(
            ant_html, add_heading_to_html(f"Beam cut data for {ant_name}:", 2)
        )

    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner("Report finished in {:.2f} seconds".format(stop - start))
    return


def main():
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.heading("Welcome to the AstroHACK BeamCut reduction pipeline")
    main_param_dict = param_init(parse(), msger)

    astrohack_stages = ["extract_holog", "extract_pointing", "beamcut", "exports"]
    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]

    if main_param_dict["processing_stage"] == "calibration":
        run_casa_calibration(main_param_dict, msger)
        main_param_dict["processing_stage"] = "extract_pointing"

    if (
        main_param_dict["parallel"]
        and main_param_dict["processing_stage"] in astrohack_stages
    ):
        client = local_client(
            cores=main_param_dict["ncores"],
            memory_limit=main_param_dict["memory_per_core"],
        )
    else:
        client = None

    if main_param_dict["processing_stage"] in astrohack_stages[:-1]:
        run_astrohack_reduction(main_param_dict, msger)

    if main_param_dict["processing_stage"] == "exports":
        run_astrohack_exports(main_param_dict, msger)
        main_param_dict["processing_stage"] = "report"

    if main_param_dict["processing_stage"] == "report":
        prepare_html_report(main_param_dict, msger)

    if client is not None:
        client.shutdown()

    pipeline_end = time.time()
    msger.heading(
        f"Beamcut processing finished in {format_duration(pipeline_end - pipeline_start)}, "
        + f"individual plots and text results saved at: {main_param_dict['exports_name']}."
        + f" Checkout the HTML report at: {main_param_dict['report_name']}."
    )
