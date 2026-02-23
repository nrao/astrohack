import contextlib
import io
import numpy as np
import xarray as xr
import xarray.testing
from PIL import Image, ImageChops

from astrohack.utils.fits import read_fits_no_checks


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


def are_fits_files_close(fits_path1, fits_path2, tol=1e-5):
    head1, data1 = read_fits_no_checks(fits_path1)
    head2, data2 = read_fits_no_checks(fits_path2)

    if are_dicts_close(head1, head2, tol=tol, ignored_keys=["DATE", "ORIGIN"]):
        return np.allclose(data1, data2, equal_nan=True, atol=tol)
    else:
        return False


def are_png_files_close(img_path1, img_path2, tol=1e-5):
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


def are_txt_files_equal(txt_path1, txt_path2, ignored_key_words=()):
    with open(txt_path1, "r") as txt_file1:
        txt1_content = txt_file1.read()
    with open(txt_path2, "r") as txt_file2:
        txt2_content = txt_file2.read()

    txt1_lines = txt1_content.splitlines()
    txt2_lines = txt2_content.splitlines()
    if len(txt1_lines) != len(txt2_lines):
        return False
    else:
        for i_line, line1 in enumerate(txt1_lines):
            line2 = txt2_lines[i_line]
            if line1.strip() == line2.strip():
                continue
            else:
                found_ignored_keywords = False
                for key_word in ignored_key_words:
                    if key_word in line1 and key_word in line2:
                        found_ignored_keywords = True
                        break
                if found_ignored_keywords:
                    continue
                else:
                    return False
        return True


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


def are_dicts_close(dict_a, dict_b, tol=1e-8, ignored_keys=None):
    """
    Compares dictionaries and returns True if data is close up to tolerance.

    :param dict_a: First dictionary
    :type dict_a: dict

    :param dict_b: Second dictionary
    :type dict_b: dict

    :param tol: Tolerance
    :type tol: float

    :param ignored_keys: Keys to be ignored in comparison
    :type ignored_keys: list, NoneType

    :return: is_close
    :rtype: bool
    """
    if ignored_keys is None:
        ignored_keys = []
    is_close = True
    b_keys = list(dict_b.keys())
    for key, a_value in dict_a.items():
        if key in ignored_keys:
            mode = "ignored"
        else:
            if key in b_keys:
                key_in_b = True
            elif str(key) in b_keys:
                key_in_b = True
                key = str(key)
            else:
                try:
                    b_type = type(b_keys[0])
                    if b_type(key) in b_keys:
                        key_in_b = True
                        key = b_type(key)
                    else:
                        key_in_b = False
                except TypeError:
                    key_in_b = False
            if key_in_b:
                b_value = dict_b[key]
                if isinstance(a_value, dict) and isinstance(b_value, dict):
                    is_close = is_close and are_dicts_close(
                        a_value, b_value, tol=tol, ignored_keys=None
                    )
                    mode = "dict"
                elif type(a_value) is not type(b_value):
                    is_close = False
                    mode = "type diff"
                elif a_value is None:
                    is_close = is_close and (b_value is None)
                    mode = "None"
                elif isinstance(a_value, (np.ndarray, float, int)):
                    mode = "nparray"
                    is_close = is_close and np.allclose(
                        a_value, b_value, equal_nan=True, rtol=tol
                    )
                elif isinstance(a_value, str):
                    mode = "str"
                    is_close = is_close and a_value == b_value
                elif isinstance(a_value, (list, tuple)):
                    if isinstance(a_value[0], (float, int)):
                        mode = "number list"
                        is_close = is_close and np.allclose(
                            np.array(a_value),
                            np.array(b_value),
                            equal_nan=True,
                            rtol=tol,
                        )
                    else:
                        mode = "obj list"
                        is_close = is_close and a_value == b_value
                else:
                    raise TypeError(f"Unrecognized type {type(a_value)}")
            else:
                mode = "missing"
                is_close = False

        if not is_close:
            print(f"Key: {key} => {mode}")
            return False
        # print(f"Key: {key} => {mode} == {is_close}")
    return is_close


def are_data_trees_close(tree_a, tree_b, tol=1e-8):
    """
    Compares data trees and returns True if data is close up to tolerance.
    :param tree_a: First data tree
    :type tree_a: xarray.DataTree

    :param tree_b: Second data tree
    :type tree_b: xarray.DataTree

    :param tol: Tolerance
    :type tol: float

    :return: is_close
    :rtype: bool
    """
    is_close = True
    if are_dicts_close(tree_a.attrs, tree_b.attrs, tol=tol):
        try:
            xarray.testing.assert_allclose(tree_a.dataset, tree_b.dataset, rtol=tol)
        except AssertionError:
            print(f"Failed datasets")
            return False
        if tree_a.is_leaf and tree_b.is_leaf:
            pass
        else:
            for key, subtree_a in tree_a.items():
                is_close = is_close and are_data_trees_close(
                    subtree_a, tree_b[key], tol
                )
                if not is_close:
                    print(f"Failed key = {key}")
                    return False
    else:
        return False
    return is_close


def add_data_folder_to_names_in_class(class_ref):
    # Add datafolder to names for execution
    for varname, varvalue in class_ref.__dict__.items():
        if isinstance(varvalue, str):
            if varname.split("_")[-1] == "name":
                setattr(class_ref, varname, f"{class_ref.data_dir}/{varvalue}")


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))
