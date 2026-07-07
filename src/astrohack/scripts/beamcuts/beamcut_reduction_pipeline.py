import argparse
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
    created_filtered_kwargs_dict,
)


def parse():
    parser = argparse.ArgumentParser(description="Beam cut reduction pipeline")

    parser.add_argument("ms_name", type=str, help="Path to the input ms to process.")

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
        "--plot-array-configuration",
        action="store_true",
        help="Plot array configuration, default is %(default)s",
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

    args = parser.parse_args()
    param_dict = create_param_dict(args)

    return param_dict


def create_param_dict(args):
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
    param_dict = {}
    param_dict.update(vars(args))

    param_dict["parallel"] = args.ncores >= 2
    if args.root_name is None:
        name_components = args.ms_name.split(".")[:-1]
        if len(name_components) == 0:
            param_dict["root_name"] = args.ms_name
        else:
            param_dict["root_name"] = ".".join(name_components)
    for identifier, extension in extensions.items():
        param_dict[f"{identifier}_name"] = param_dict["root_name"] + extension

    if args.antenna != "all":
        param_dict["antenna"] = args.antenna.split(",")
    if args.spectral_window != "all":
        param_dict["spectral_window"] = [
            int(spw_id) for spw_id in args.spectral_window.split(",")
        ]
    if args.exclude_bad_antennas is not None:
        param_dict["exclude_bad_antennas"] = args.exclude_bad_antennas.split(",")

    initialization_check(param_dict, "Beam cut reduction parameters")
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spectral_window"]
    param_dict["exclude_antennas"] = param_dict["exclude_bad_antennas"]
    return param_dict


def execute_step(param_dict, function, next_stage, msger):
    function_name = function.__name__
    if param_dict["processing_stage"] == function_name:
        try:
            msger.one_liner(f"Executing {function_name}...")
            function(**created_filtered_kwargs_dict(param_dict, function))
            msger.done()
            param_dict["processing_stage"] = next_stage
            return True, None
        except Exception as the_exception:
            return False, the_exception
    else:
        return True, None


def data_reduction(param_dict, msger):
    status = True
    exec_exception = None
    exec_list = [
        ["extract_holog", extract_pointing],
        ["beamcut", extract_holog],
        ["plotting", beamcut],
    ]
    for next_stage, function in exec_list:
        if status:
            status, exec_exception = execute_step(
                param_dict, function, next_stage, msger
            )

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def post_processing(param_dict, msger):
    param_dict["destination"] = param_dict["exports_name"]
    bmc_mds = open_beamcut(param_dict["beamcut_name"])
    if bmc_mds is not None:
        msger.heading("Producing pipeline exports...")
        beamcut_methods = [
            bmc_mds.plot_in_amplitude,
            bmc_mds.plot_in_phase,
            bmc_mds.plot_in_db,
            bmc_mds.plot_lm_offsets,
            bmc_mds.export_report,
        ]
        for method in beamcut_methods:
            msger.one_liner(f"Running {method.__name__}...")
            method(**created_filtered_kwargs_dict(param_dict, method))
        msger.one_liner("Beamcut exports Done!")

    if param_dict["plot_array_configuration"] or param_dict["plot_pointing"]:
        pnt_mds = open_pointing(param_dict["point_name"])
        if pnt_mds is not None:
            msger.heading("Producing pointing exports...")
            pnt_methods = []
            if param_dict["plot_array_configuration"]:
                pnt_methods.append(pnt_mds.plot_array_configuration)
            if param_dict["plot_pointing"]:
                pnt_methods.append(pnt_mds.plot_pointing_in_time)
            for method in pnt_methods:
                msger.one_liner(f"Running {method.__name__}...")
                method(**created_filtered_kwargs_dict(param_dict, method))
            msger.one_liner("Pointing exports Done!")
    return


def main():
    msger = MessageBoard()
    msger.heading("Welcome to AstroHACK BeamCut Pipeline")
    main_param_dict = parse()

    if main_param_dict["parallel"]:
        client = local_client(
            cores=main_param_dict["ncores"],
            memory_limit=main_param_dict["memory_per_core"],
        )
    else:
        client = None

    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]
    data_reduction(main_param_dict, msger)

    if main_param_dict["processing_stage"] == "plotting":
        post_processing(main_param_dict, msger)

    if main_param_dict["parallel"]:
        client.shutdown()

    msger.heading("Beamcut pipeline complete!")
