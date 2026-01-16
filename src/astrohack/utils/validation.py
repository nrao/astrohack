from PIL import Image, ImageChops
import contextlib, io
import numpy as np

from astrohack.antenna.antenna_surface import SUPPORTED_POL_STATES
from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT
from astrohack.utils import trigo_units, length_units, time_units, freq_units
from astrohack.utils import possible_splits
from astrohack.visualization.plot_tools import astrohack_cmaps


def custom_plots_checker(allowed_type):
    if allowed_type == "colormaps":
        return astrohack_cmaps
    elif "split" in allowed_type:
        return custom_split_checker(allowed_type)
    elif "units" in allowed_type:
        return custom_unit_checker(allowed_type)
    else:
        return "Not found"


def custom_unit_checker(unit_type):
    if unit_type == "units.trig":
        return trigo_units

    elif unit_type == "units.length":
        return length_units

    elif unit_type == "units.time":
        return time_units

    elif unit_type == "units.frequency":
        return freq_units
    elif unit_type == "units.radec":
        valid_units = trigo_units.copy()
        valid_units.append("radec")
        return valid_units
    else:
        return "Not found"


def custom_split_checker(split_type):
    if split_type == "split.complex":
        return possible_splits
    else:
        return "Not found"


def custom_panel_checker(check_type):
    if check_type == "panel.models":
        return PANEL_MODEL_DICT.keys()
    elif check_type == "panel.pol_states":
        return SUPPORTED_POL_STATES
    else:
        return "Not found"


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


def are_png_files_equal(img_path1, img_path2):
    try:
        # Open images (Pillow handles various modes and removes metadata concerns for pixel data)
        with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
            # Ensure both images are in the same mode for a reliable comparison (e.g., 'RGBA')
            img1 = img1.convert("RGBA")
            img2 = img2.convert("RGBA")

            # Check if dimensions are the same
            if img1.size != img2.size:
                return False

            # Calculate the difference between the images
            # This results in a new image where differing pixels are non-zero
            diff = ImageChops.difference(img1, img2)

            return np.allclose(diff, 0, atol=1e-5)

    except IOError as e:
        print(f"Error opening images: {e}")
        return False


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
