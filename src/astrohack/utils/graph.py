import shutil
import pathlib
import dask
import xarray as xr
import toolviper.utils.logger as logger
import copy
import zarr
import glob

from astrohack.utils.text import approve_prefix
from astrohack.utils.text import param_to_list


def _factorized_graph_execution_return(status, ret_list, fetch_ret):
    if fetch_ret:
        if status:
            return status, ret_list
        else:
            return status, None
    else:
        return status


def _consolidate_a_level(key_path):
    if pathlib.Path(key_path).is_dir():
        key = key_path.split("/")[-1]
        this_lvl_xdt = xr.DataTree(name=key)
        this_lvl_xdt.to_zarr(key_path)
        del this_lvl_xdt
        this_zarr_group = zarr.open(key_path, mode="r+")
        zarr.convenience.consolidate_metadata(this_zarr_group.store)
    else:
        logger.warning(f"There is an unexpected entity at {key_path}")


def _consolidate_output_mds(key_order, output_mds):
    mds_path = output_mds.filename
    logger.info(f"Consolidating {mds_path}...")

    # Hardcoded number of levels of extract_holog products as they are 3 leveled but execution is 2 leveled.
    if output_mds.root.attrs["origin_info"]["creator_function"] == "extract_holog":
        n_lvls = 3
    elif output_mds.root.attrs["origin_info"]["creator_function"] == "combine":
        n_lvls = 2
    else:
        n_lvls = len(key_order)

    if n_lvls == 1:
        pass
    elif n_lvls == 2 or n_lvls == 3:
        lvl_0_list = glob.glob(f"{mds_path}/*")
        for key_path_0 in lvl_0_list:
            if n_lvls == 3:
                lvl_1_list = glob.glob(f"{key_path_0}/*")
                for key_path_1 in lvl_1_list:
                    _consolidate_a_level(key_path_1)
            _consolidate_a_level(key_path_0)
    else:
        raise NotImplementedError(f"Unsupported number of levels: {n_lvls}")

    root_group = zarr.open(mds_path, mode="r+")  # Open in read/write mode
    zarr.convenience.consolidate_metadata(root_group.store)

    output_mds.open()


def _construct_general_graph_recursively(
    looping_dict,
    chunk_function,
    param_dict,
    delayed_list,
    key_order,
    output_mds,
    parallel,
    oneup=None,
):
    if len(key_order) == 0:
        if isinstance(looping_dict, xr.DataTree):
            param_dict["xdt_data"] = looping_dict
        elif isinstance(looping_dict, xr.Dataset):
            param_dict["xds_data"] = looping_dict
        elif isinstance(looping_dict, dict):
            param_dict["dic_data"] = looping_dict
        else:
            param_dict["unk_data"] = looping_dict

        if output_mds is None:
            args = [param_dict]
        else:
            args = [param_dict, output_mds]
        if parallel:
            delayed_list.append(dask.delayed(chunk_function)(*args))
        else:
            delayed_list.append((chunk_function, args))
    else:
        key = key_order[0]

        exec_list = param_to_list(param_dict[key], looping_dict, key)
        white_list = [key for key in exec_list if approve_prefix(key)]

        for item in white_list:
            this_param_dict = copy.deepcopy(param_dict)
            this_param_dict[f"this_{key}"] = item

            if item in looping_dict:
                _construct_general_graph_recursively(
                    looping_dict=looping_dict[item],
                    chunk_function=chunk_function,
                    param_dict=this_param_dict,
                    delayed_list=delayed_list,
                    key_order=key_order[1:],
                    output_mds=output_mds,
                    parallel=parallel,
                    oneup=item,
                )

            else:
                if oneup is None:
                    logger.warning(f"{item} is not present in looping dict")
                else:
                    logger.warning(f"{item} is not present for {oneup}")


def create_and_execute_graph_from_dict(
    looping_dict,
    chunk_function,
    param_dict,
    key_order,
    output_mds=None,
    parallel=False,
    fetch_returns=False,
):

    if hasattr(looping_dict, "root"):
        looping_dict = looping_dict.root

    # List created here to avoid complicated returns due to recursion.
    delayed_list = []
    _construct_general_graph_recursively(
        looping_dict=looping_dict,
        chunk_function=chunk_function,
        param_dict=param_dict,
        delayed_list=delayed_list,
        key_order=key_order,
        output_mds=output_mds,
        parallel=parallel,
    )

    if len(delayed_list) == 0:
        logger.warning(f"List of delayed processing jobs is empty: No data to process")
        return _factorized_graph_execution_return(False, [], fetch_returns)

    if parallel:
        return_list = dask.compute(delayed_list)[0]
    else:
        return_list = []
        for function, args in delayed_list:
            return_list.append(function(*args))

    if output_mds is not None:
        if len(output_mds.keys()) == 0:
            logger.warning("Processing did not yield any data")
            return _factorized_graph_execution_return(False, return_list, fetch_returns)
        else:
            return _factorized_graph_execution_return(True, return_list, fetch_returns)

    return _factorized_graph_execution_return(True, return_list, fetch_returns)


def create_and_execute_graph_from_dict_2(
    looping_dict,
    chunk_function,
    param_dict,
    key_order,
    output_mds=None,
    parallel=False,
    fetch_returns=False,
):
    if hasattr(looping_dict, "root"):
        looping_dict = looping_dict.root

    if output_mds is not None:
        output_mds.write(mode="a")

    # List created here to avoid complicated returns due to recursion.
    delayed_list = []
    _construct_general_graph_recursively(
        looping_dict=looping_dict,
        chunk_function=chunk_function,
        param_dict=param_dict,
        delayed_list=delayed_list,
        key_order=key_order,
        output_mds=output_mds,
        parallel=parallel,
    )

    if len(delayed_list) == 0:
        logger.warning(f"List of delayed processing jobs is empty: No data to process")
        return _factorized_graph_execution_return(False, [], fetch_returns)

    if parallel:
        return_list = dask.compute(delayed_list)[0]
    else:
        return_list = []
        for function, args in delayed_list:
            return_list.append(function(*args))

    if output_mds is not None:
        _consolidate_output_mds(key_order, output_mds)

        if len(output_mds.keys()) == 0:
            logger.warning("Processing did not yield any data")
            shutil.rmtree(output_mds.filename)
            return _factorized_graph_execution_return(False, return_list, fetch_returns)
        else:

            return _factorized_graph_execution_return(True, return_list, fetch_returns)

    return _factorized_graph_execution_return(True, return_list, fetch_returns)


def compute_graph_from_lists(
    param_dict, chunk_function, looping_key_list, parallel=False
):
    """
    Creates and executes a graph based on entries in a parameter dictionary that are lists
    Args:
        param_dict: The parameter dictionary
        chunk_function: The function for the operation chunk
        looping_key_list: The keys that are lists in the parameter dictionaries over which to loop over
        parallel: execute graph in parallel?

    Returns:
        A list containing the returns of the calls to the chunk function.
    """
    niter = len(param_dict[looping_key_list[0]])

    delayed_list = []
    result_list = []
    for i_iter in range(niter):
        this_param = copy.deepcopy(param_dict)
        for key in looping_key_list:
            this_param[f"this_{key}"] = param_dict[key][i_iter]

        if parallel:
            delayed_list.append(dask.delayed(chunk_function)(dask.delayed(this_param)))
        else:
            delayed_list.append(0)
            result_list.append(chunk_function(this_param))

    if parallel:
        result_list = dask.compute(delayed_list)

    return result_list
