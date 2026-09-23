import glob
import time

from toolviper.dask.client import local_client

from astrohack import (
    extract_pointing,
    extract_holog,
    holog,
    combine,
    panel,
    open_pointing,
    open_holog,
    open_image,
    open_panel,
)
from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    basic_holography_parser,
    common_parameter_initialization_for_holography,
    run_astrohack_function,
    open_astrohack_file,
    add_basic_info_and_parameters_to_report,
)
from astrohack.utils.text import (
    lnbr,
    add_heading_to_html,
    create_single_html_image_with_header,
    create_html_file_from_body,
)


def parse(pipeline_type: str, stages: list):
    parser = basic_holography_parser(pipeline_type, stages)

    # Extra holography options come here
    parser.add_argument(
        "-c",
        "--combine",
        action="store_true",
        help="Combine SPWs to improve SNR (all selected SPWs will be combined)",
    )

    parser.add_argument(
        "-z",
        "--use-zernike-phase-fitting",
        action="store_true",
        help=(
            "Use zernike phase fitting instead of regular perturbation phase fitting "
            "(ngVLA uses this even when option is not given)."
        ),
    )

    return vars(parser.parse_args())


def param_init(pipeline_type: str, msger: MessageBoard):
    extensions = {
        "delay_cal": ".dcal",
        "bandpass_cal": ".bcal",
        "gain_cal": ".gcal",
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "image": ".image.zarr",
        "combine": ".combine.zarr",
        "panel": ".panel.zarr",
        "exports": ".exports",
        "report": "-report.html",
    }
    stages = [
        "calibration",
        "extract_pointing",
        "extract_holog",
        "holog",
        "panel",
        "exports",
        "report",
    ]
    param_dict = common_parameter_initialization_for_holography(
        pipeline_type, stages, extensions, parse, msger
    )

    # extra parameter initialization comes here

    initialization_check(
        param_dict, f"{pipeline_type.capitalize()} reduction parameters"
    )
    return param_dict, stages


def run_casa_calibration(param_dict: dict, msger: MessageBoard):
    msger.heading("Calibration will come here!")
    return


def run_astrohack_reduction(param_dict: dict, msger: MessageBoard):
    # Astrohack convenience changes
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spw"]
    param_dict["exclude_antennas"] = param_dict["exclude_bad_antennas"]
    param_dict["ms_name"] = param_dict["msname"]

    status = True
    exec_exception = None

    exec_list = [
        ["extract_holog", extract_pointing],
        ["holog", extract_holog],
        ["panel", holog],
        ["exports", panel],
    ]

    for next_stage, function in exec_list:
        if status and param_dict["processing_stage"] == function.__name__:
            status, exec_exception = run_astrohack_function(
                param_dict,
                function,
                msger,
            )
            if param_dict["processing_stage"] == "holog" and param_dict["combine"]:
                status, exec_exception = run_astrohack_function(
                    param_dict,
                    combine,
                    msger,
                )
                if status:
                    param_dict["image_name"] = param_dict["combine_name"]
            if status:
                param_dict["processing_stage"] = next_stage

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def run_astrohack_exports(param_dict: dict, msger: MessageBoard):
    param_dict["destination"] = param_dict["exports_name"]
    pnt_mds = open_astrohack_file(open_pointing, param_dict["point_name"])
    # hlg_mds = open_astrohack_file(open_holog, param_dict["holog_name"])
    img_mds = open_astrohack_file(open_image, param_dict["image_name"])
    pnl_mds = open_astrohack_file(open_panel, param_dict["panel_name"])

    export_methods = [
        pnt_mds.plot_array_configuration,
        img_mds.plot_beams,
        img_mds.export_phase_fit_results,
        img_mds.export_zernike_fit_results,
        img_mds.plot_zernike_model,
        pnl_mds.plot_antennas,
        pnl_mds.export_screws,
        pnl_mds.export_gain_tables,
    ]
    if param_dict["plot_pointing"]:
        param_dict["plot_antennas_separately"] = True
        export_methods.append(pnt_mds.plot_pointing_in_time)

    for export_method in export_methods:
        status, exec_exception = run_astrohack_function(
            param_dict, export_method, msger
        )
        if not status:
            raise RuntimeError(
                f"{export_method.__name__} failed see above for details."
            ) from exec_exception

    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    pnl_mds = open_astrohack_file(open_panel, param_dict["panel_name"])
    msger.one_liner("Preparing report...")
    start = time.time()
    exports_name = param_dict["exports_name"]
    report_title = f"Holography report for {param_dict['filename']}"
    html_body = add_heading_to_html(report_title, 1)
    html_body += add_basic_info_and_parameters_to_report(param_dict)
    html_body += create_single_html_image_with_header(
        f"{exports_name}/point_array_configuration.png",
        "Array configuration during observation",
        heading_level=2,
    )
    html_body += f"{lnbr}<br>{lnbr}"

    for ant_key, ant_xds in pnl_mds.items():
        ant_name = ant_key.split("_")[1]
        for ddi_key in ant_xds.keys():
            ddi_name = ddi_key.split("_")[1]
            # summary to be done here per antenna/ddi
            summ_name = f"{exports_name}/obs_summary_{ant_key}_{ddi_key}.txt"
            pnl_mds.observation_summary(
                summary_file=summ_name,
                ant=ant_name,
                ddi=ddi_name,
                parallel=False,
                print_summary=False,
            )

    create_html_file_from_body(html_body, report_title, param_dict["report_name"])
    stop = time.time()
    msger.one_liner("Report finished in {:.2f} seconds".format(stop - start))
    return


def main():
    pipeline_type = "holography"
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.welcome_message(pipeline_type)

    main_param_dict, stages = param_init(pipeline_type, msger)

    astrohack_stages = stages[1:6]
    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]

    if main_param_dict["processing_stage"] == stages[0]:
        run_casa_calibration(main_param_dict, msger)
        main_param_dict["processing_stage"] = astrohack_stages[0]

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

    if main_param_dict["processing_stage"] == astrohack_stages[-1]:
        run_astrohack_exports(main_param_dict, msger)
        main_param_dict["processing_stage"] = stages[-1]

    if main_param_dict["processing_stage"] == stages[-1]:
        prepare_html_report(main_param_dict, msger)

    if client is not None:
        client.shutdown()

    pipeline_end = time.time()
    msger.goodbye_message(pipeline_type, main_param_dict, pipeline_end - pipeline_start)
