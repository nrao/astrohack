import time

from toolviper.dask.client import local_client

from astrohack import (
    extract_pointing,
    extract_holog,
    holog,
    combine,
    panel,
)
from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    basic_holography_parser,
    common_parameter_initialization_for_holography,
    run_astrohack_function,
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
    msger.heading("Exports will come here!")
    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    msger.heading("Report will come here!")
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
