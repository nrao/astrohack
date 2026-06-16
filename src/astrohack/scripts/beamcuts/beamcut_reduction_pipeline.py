import argparse
from toolviper.dask.client import local_client
from astrohack import (
    extract_pointing,
    extract_holog,
    beamcut,
    open_pointing,
    open_beamcut,
)
from astrohack.utils.user_interaction import initialization_check
import inspect


def create_param_dict(args):
    extensions = {
        "point": ".point.zarr",
        "holog": ".holog.zarr",
        "beamcut": ".beamcut.zarr",
        "exports": ".exports",
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

    initialization_check(param_dict, "Beam cut reduction parameters")
    param_dict["ant"] = param_dict["antenna"]
    param_dict["ddi"] = param_dict["spectral_window"]
    return param_dict


def parse():
    parser = argparse.ArgumentParser(description="Beam cut reduction pipeline")

    parser.add_argument("ms_name", type=str, help="Path to the input ms to process.")

    parser.add_argument(
        "-r",
        "--root-name",
        type=str,
        default=None,
        help="Root name for the products of the pipeline, default"
        " is ms_name without extension",
    )

    parser.add_argument(
        "-s",
        "--spectral-window",
        type=str,
        default="all",
        help="Select SPWs for which to produce beamcuts, for a list of"
        "SPWs use comma separated integers with no spaces, e.g.:"
        "'0,1,2', default is %(default)s",
    )

    parser.add_argument(
        "-a",
        "--antenna",
        type=str,
        default="all",
        help="Select antennas for which to produce beamcuts, "
        "for a list of antennas use comma separated names with"
        " no spaces, e.g.: 'ea01,ea02', default is %(default)s",
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
        default="extract_pointing",
        choices=["extract_pointing", "extract_holog", "beamcut", "plotting"],
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

    args = parser.parse_args()
    param_dict = create_param_dict(args)

    return param_dict


def created_filtered_kwargs_dict(param_dict, function):
    valid_kwarg_keys = inspect.signature(function).parameters
    filtered_dict = {
        key: value for key, value in param_dict.items() if key in valid_kwarg_keys
    }
    return filtered_dict


def execute_step(param_dict, label, function, next_stage):
    function_name = function.__name__
    if param_dict["processing_stage"] == function_name:
        try:
            print("\n" + 40 * "#" + "\n")
            print(f"Executing {function_name}...")
            function(**created_filtered_kwargs_dict(param_dict, function))
            print(f"{function_name.capitalize()} done!")
            param_dict["processing_stage"] = next_stage
            return True, None
        except Exception as the_exception:
            return False, the_exception
    else:
        return True, None


def data_reduction(param_dict):
    status = True
    exec_exception = None
    exec_list = [
        ["point", "extract_holog", extract_pointing],
        ["holog", "beamcut", extract_holog],
        ["beamcut", "plotting", beamcut],
    ]
    for label, next_stage, function in exec_list:
        if status:
            status, exec_exception = execute_step(
                param_dict, label, function, next_stage
            )

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exec_exception

    return


def post_processing(param_dict):
    param_dict["destination"] = param_dict["exports_name"]
    bmc_mds = open_beamcut(param_dict["beamcut_name"])
    if bmc_mds is not None:
        print("\n" + 40 * "#" + "\n")
        beamcut_methods = [
            bmc_mds.plot_beamcut_in_amplitude,
            bmc_mds.plot_beamcut_in_phase,
            bmc_mds.plot_beamcut_in_attenuation,
            bmc_mds.plot_beam_cuts_over_sky,
            bmc_mds.create_beam_fit_report,
        ]
        for method in beamcut_methods:
            print(f"Running {method.__name__}...")
            method(**created_filtered_kwargs_dict(param_dict, method))
        print("Beamcut exports Done!\n")

    if param_dict["plot_array_configuration"] or param_dict["plot_pointing"]:
        pnt_mds = open_pointing(param_dict["point_name"])
        if pnt_mds is not None:
            print("\n" + 40 * "#" + "\n")
            pnt_methods = []
            if param_dict["plot_array_configuration"]:
                pnt_methods.append(pnt_mds.plot_array_configuration)
            if param_dict["plot_pointing"]:
                pnt_methods.append(pnt_mds.plot_pointing_in_time)
            for method in pnt_methods:
                print(f"Running {method.__name__}...")
                method(**created_filtered_kwargs_dict(param_dict, method))
            print("Pointing exports Done!\n")
    return


def main():
    main_param_dict = parse()

    if main_param_dict["parallel"]:
        client = local_client(
            cores=main_param_dict["ncores"],
            memory_limit=main_param_dict["memory_per_core"],
        )
    else:
        client = None

    main_param_dict["processing_stage"] = main_param_dict["starting_stage"]
    data_reduction(main_param_dict)

    if main_param_dict["processing_stage"] == "plotting":
        post_processing(main_param_dict)

    if main_param_dict["parallel"]:
        client.shutdown()
