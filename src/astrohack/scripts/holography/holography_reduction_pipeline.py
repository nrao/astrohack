import time

from toolviper.dask.client import local_client

from astrohack.utils.pipeline_support import MessageBoard


def parse():
    param_dict = {}
    return param_dict


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
