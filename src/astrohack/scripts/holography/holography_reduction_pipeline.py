import time

from toolviper.dask.client import local_client

from astrohack.utils.pipeline_support import (
    MessageBoard,
    create_parser_with_base_options,
    list_input_tooltip,
)


def parse(pipeline_type: str, stages: list):
    parser = create_parser_with_base_options(pipeline_type, stages)

    parser.add_argument(
        "-d",
        "--data-column",
        type=str,
        default="CORRECTED_DATA",
        help="Data column to be extracted from MS, default is %(default)s",
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


def fetch_ms_data():
    return


def param_init(param_dict: dict, msger: MessageBoard):

    return param_dict


def run_casa_calibration(param_dict: dict, msger: MessageBoard):
    return


def run_astrohack_reduction(param_dict: dict, msger: MessageBoard):
    return


def run_astrohack_exports(param_dict: dict, msger: MessageBoard):
    return


def prepare_html_report(param_dict: dict, msger: MessageBoard):
    return


def main():
    pipeline_type = "holography"
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.welcome_message(pipeline_type)
    stages = [
        "calibration",
        "extract_pointing",
        "extract_holog",
        "holog",
        "panel",
        "exports",
        "report",
    ]

    main_param_dict = param_init(parse(pipeline_type, stages), msger)

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
