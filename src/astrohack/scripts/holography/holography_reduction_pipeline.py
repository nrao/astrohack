import time

from toolviper.dask.client import local_client

from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    basic_holography_parser,
    common_parameter_initialization_for_holography,
)


def parse(pipeline_type: str, stages: list):
    parser = basic_holography_parser(pipeline_type, stages)

    # Extra holography options come here

    return vars(parser.parse_args())


def param_init(pipeline_type: str, msger: MessageBoard):
    extensions = {
        "delay_cal": ".dcal",
        "bandpass_cal": ".bcal",
        "gain_cal": ".gcal",
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "image": ".image.zarr",
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
    msger.heading("Reduction will come here!")
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
