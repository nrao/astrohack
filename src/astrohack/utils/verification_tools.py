import contextlib
import io

import numpy as np
import xarray as xr
from PIL import Image, ImageChops


def are_lists_equal(list_a, list_b):
    n_a = len(list_a)
    n_b = len(list_b)
    if n_a != n_b:
        return False
    else:
        equal = True
        for item in list_a:
            equal = equal and item in list_b
        return equal


def are_png_files_equal(img_path1, img_path2, tol=1e-5):
    try:
        # Open images (Pillow handles various modes and removes metadata concerns for pixel data)
        with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
            # Ensure both images are in the same mode for a reliable comparison (e.g., 'RGBA')
            img1 = img1.convert("RGBA")
            img2 = img2.convert("RGBA")

            # Check if dimensions are the same
            if img1.size != img2.size:
                return False, f"PNG sizes differ"

            # Calculate the difference between the images
            # This results in a new image where differing pixels are non-zero
            diff = ImageChops.difference(img1, img2)
            mean_diff = np.mean(diff)
            return np.abs(mean_diff) < tol, "Mean diff: {np.mean(np.absolute(diff))}"

    except IOError as e:
        print(f"Error opening images: {e}")
        return False, f"Failed opening images"


def capture_prints_from_function(function, args=None):
    # Use redirect_stdout to capture the function's output
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        if args is None:
            function()
        else:
            function(*args)

    # Get the captured output as a string
    return output_capture.getvalue()


def dump_captured_output_to_file(function, dump_file, args=None):
    output_captured = capture_prints_from_function(function, args)
    with open(dump_file, "w") as dump_file_obj:
        dump_file_obj.write(output_captured)


def are_txt_files_equal(txt_path1, txt_path2):
    with open(txt_path1, "r") as txt_file1:
        txt1_content = txt_file1.read()
        with open(txt_path2, "r") as txt_file2:
            txt2_content = txt_file2.read()
            return txt1_content == txt2_content


def is_captured_output_equal_to_txt_reference(function, txt_ref, args=None):
    captured_output = capture_prints_from_function(function, args)
    with open(txt_ref, "r") as ref_file:
        ref_content = ref_file.read()
    return ref_content == captured_output


def _get_ds_metadata(ds):
    if hasattr(ds, "_input_pars"):
        metadata = getattr(ds, "_input_pars")
    elif isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataTree):
        metadata = getattr(ds, "attrs")
    else:
        metadata = ds.root.attrs
    return metadata


def _compare_dictionaries(dict_a, dict_b, metaname):
    different_keys = False
    for key in dict_a.keys():
        if key not in dict_b.keys():
            different_keys = True

    if different_keys:
        return True, f"{metaname} keys do not match"

    different_values = False
    for key, value in dict_a.items():
        if isinstance(value, np.ndarray) or isinstance(value, xr.DataArray):
            if isinstance(value, np.ndarray):
                value_a = value
                value_b = dict_b[key]
            else:
                value_a = value.values
                value_b = dict_b[key].values
            is_str_arr = value_a.dtype.char in ["U", "S", "O"]
            if is_str_arr:
                different_values = np.any(value_a != value_b)
            else:
                different_values = not np.allclose(value_a, value_b, equal_nan=True)
        else:
            different_values = value != dict_b[key]
    if different_values:
        return True, f"{metaname} values do not match"

    return False, ""


def _is_mds_metadata_different(mds_a, mds_b, metaname="metadata"):
    metadata_a = _get_ds_metadata(mds_a)
    metadata_b = _get_ds_metadata(mds_b)
    return _compare_dictionaries(metadata_a, metadata_b, metaname)


def _is_xdtree(data):
    return hasattr(data, "is_leaf")


def _get_xds_data_from_dict(data):
    if isinstance(data, xr.Dataset):
        return data
    elif _is_xdtree(data):
        if data.is_leaf:
            return data.ds
        else:
            return False
    else:
        return False


def _compare_xds_data(xds_a, xds_b, label):
    metadata_different, msg = _is_mds_metadata_different(
        xds_a, xds_b, metaname=f"{label} attribute"
    )
    if metadata_different:
        return True, msg

    else:
        return _compare_dictionaries(xds_a, xds_b, metaname=f"{label} data variable")


def _are_data_dicts_different(dict_a, dict_b, label=""):
    xds_a = _get_xds_data_from_dict(dict_a)
    xds_b = _get_xds_data_from_dict(dict_b)
    compare_dict = not (xds_a or xds_b)

    if compare_dict:
        for key in dict_a.keys():
            if key not in dict_b.keys():
                return True, f"{label[2:]} keys do not match"
            elif "info" in key:
                return _compare_dictionaries(dict_a[key], dict_b[key], label)
            else:
                data_dicts_are_different, msg = _are_data_dicts_different(
                    dict_a[key], dict_b[key], label=f"{label}, {key}"
                )
                if data_dicts_are_different:
                    return True, msg
    else:
        return _compare_xds_data(xds_a, xds_b, label=label[2:])

    return False, ""


def mds_equality_test(mds_a, mds_b):
    """
    :param mds_a: First MDS object
    :param mds_b: Second MDS object
    :return: Equality test result, error message.
    """

    metadata_different, msg = _is_mds_metadata_different(mds_a, mds_b)
    if metadata_different:
        return False, f"{mds_a.filename} and {mds_b.filename} {msg}."

    data_dicts_different, msg = _are_data_dicts_different(mds_a, mds_b)
    if data_dicts_different:
        return False, f"{mds_a.filename} and {mds_b.filename} {msg}."

    return True, f"{mds_a.filename} and {mds_b.filename} are equal"


def add_data_folder_to_names_in_class(class_ref):
    # Add datafolder to names for execution
    for varname, varvalue in class_ref.__dict__.items():
        if isinstance(varvalue, str):
            if varname.split("_")[-1] == "name":
                setattr(class_ref, varname, f"{class_ref.data_dir}/{varvalue}")
