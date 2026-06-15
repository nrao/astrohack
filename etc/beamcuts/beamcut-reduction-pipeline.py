import argparse
import numpy as np
from toolviper.dask.client import local_client
from astrohack import extract_pointing, extract_holog, beamcut, open_pointing
from astrohack.utils.user_interaction import yesno
from pathlib import Path


def create_param_dict(args):
    extensions = {
        "pnt": ".point.zarr",
        "hlg": ".holog.zarr",
        "bmc": ".beamcut.zarr",
        "plt": ".plots",
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

    print("Input parameters:")
    print_dict_simple(param_dict)
    print()
    if not param_dict["assume_yes"]:
        if not yesno("Proceed?"):
            exit(0)

    return param_dict


def print_dict_simple(the_dict, ident=4):
    key_len = 0
    for key in the_dict.keys():
        if len(key) > key_len:
            key_len = len(key)

    for key, value in the_dict.items():
        print(f"{ident*' '}{key:{key_len}s} => {value}")


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

    args = parser.parse_args()
    param_dict = create_param_dict(args)

    return param_dict


def execute_step(param_dict, label, function, kwargs, next_stage):
    function_name = function.__name__
    if (
        not Path(param_dict[f"{label}_name"]).is_dir() or param_dict["overwrite"]
    ) and param_dict["processing_stage"] == function_name:
        try:
            print(f"Executing {function_name}...")
            function(**kwargs)
            print(f"{function_name.capitalize()} done!")
            param_dict["processing_stage"] = next_stage
            return True, None
        except Exception as the_exception:
            return False, the_exception
    else:
        return True, None


def data_reduction(param_dict):
    status, exception = execute_step(
        param_dict,
        "pnt",
        extract_pointing,
        {
            "ms_name": param_dict["ms_name"],
            "point_name": param_dict["pnt_name"],
            "parallel": param_dict["parallel"],
            "overwrite": param_dict["overwrite"],
        },
        "extract_holog",
    )

    if status:
        status, exception = execute_step(
            param_dict,
            "hlg",
            extract_holog,
            {
                "ms_name": param_dict["ms_name"],
                "point_name": param_dict["pnt_name"],
                "holog_name": param_dict["hlg_name"],
                "ant": param_dict["antenna"],
                "ddi": param_dict["spectral_window"],
                "data_column": param_dict["data_column"],
                "parallel": param_dict["parallel"],
                "overwrite": param_dict["overwrite"],
            },
            "beamcut",
        )

    if status:
        status, exception = execute_step(
            param_dict,
            "bmc",
            beamcut,
            {
                "holog_name": param_dict["hlg_name"],
                "beamcut_name": param_dict["bmc_name"],
                "ant": param_dict["antenna"],
                "ddi": param_dict["spectral_window"],
                "parallel": param_dict["parallel"],
                "overwrite": param_dict["overwrite"],
            },
            "plotting",
        )

    if not status:
        raise RuntimeError(
            f"{param_dict['processing_stage']} failed, see above for details."
        ) from exception

    return


def post_processing():
    return


if __name__ == "__main__":
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

    post_processing()

    if main_param_dict["parallel"]:
        client.shutdown()
