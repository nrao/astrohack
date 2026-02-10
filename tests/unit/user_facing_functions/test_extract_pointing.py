import os
import pathlib
import shutil
import pytest
import toolviper

from astrohack.extract_pointing import extract_pointing
from astrohack.utils.verification_tools import add_data_folder_to_names_in_class


class TestExtractPointing:
    data_dir = "point_data"
    ms_name = "ea25_cal_small_before_fixed.split.ms"
    def_pnt_name = "ea25_cal_small_before_fixed.split.point.zarr"
    alt_pnt_name = "ea25_short.point.zarr"
    ref_pnt_name = "ea25_cal_small_before_reference.point.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class"""
        toolviper.utils.data.download(file=cls.ms_name, folder=cls.data_dir)
        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a setup_class."""
        # shutil.rmtree(cls.data_dir)

    def test_extract_pointing_defaults(self):
        """Test extract_pointing with default parameters"""
        new_pnt_mds = extract_pointing(ms_name=self.ms_name, overwrite=True)
        assert pathlib.Path(self.def_pnt_name).is_dir()


# class TestExtractPointing:
#     data_dir = "point_data"
#     ms_name = "ea25_cal_small_before_fixed.split.ms"
#     def_pnt_name = "ea25_cal_small_before_fixed.split.point.zarr"
#     alt_pnt_name = "ea25_short.point.zarr"
#     ref_pnt_name = "ea25_cal_small_before_reference.point.zarr"
#     print("repolho")

# @classmethod
# def setup_class(cls):
#     """setup any state specific to the execution of the given test class"""
#     print("feijao")
#     toolviper.utils.data.download(file=cls.ms_name, folder=cls.data_dir)
#     # add_data_folder_to_names_in_class(cls)
#
# @classmethod
# def teardown_class(cls):
#     """teardown any state that was previously setup with a call to setup_class"""
#     return
#     shutil.rmtree(cls.data_dir)

# def test_extract_pointing_default(self):
#     """Test extract_pointing with default parameters"""
#     print("aaaaa")
#     print(self.ms_name)
#     # point_obj = extract_pointing(ms_name=self.ms_name)
#     # print(point_obj)
#     assert True
#     # # Check the keys of the returned dictionary
#     # expected_keys = ["point_meta_ds", "ant_ea04", "ant_ea06", "ant_ea25"]
#     #
#     # for key in point_obj.keys():
#     #     assert key in expected_keys

# @pytest.mark.skip(reason="mds equality test is not yet robust")
# def test_extract_pointing_point_name(self):
#     """Test extract_pointing and save to given point name"""
#     point_name = os.path.join(self.data_dir, "test_user_point_name.zarr")
#     point_obj = extract_pointing(ms_name=self.ms_name, point_name=point_name)
#
#     assert os.path.exists(point_name)
#
#     # Check that the returned dictionary contains the given point_name
#     assert point_obj.file == point_name
#
# @pytest.mark.skip(reason="mds equality test is not yet robust")
# def test_extract_pointing_overwrite_true(self):
#     """Test extract_pointing and overwrite existing pointing file"""
#     point_name = os.path.join(self.data_dir, "test_user_overwrite.zarr")
#     extract_pointing(ms_name=self.ms_name, point_name=point_name)
#     initial_time = os.path.getctime(point_name)
#
#     extract_pointing(ms_name=self.ms_name, point_name=point_name, overwrite=True)
#     modified_time = os.path.getctime(point_name)
#
#     assert initial_time != modified_time
#
# @pytest.mark.skip(reason="mds equality test is not yet robust")
# def test_extract_pointing_invalid_ms_name(self):
#     """Test extract_pointing and check that invalid_ms does not create point file"""
#     # Exceptions are not raised by the code, therefore doing a silly check here
#     try:
#         # Changed this because it does fail if you give a non-existent ms. So we will check that.
#         extract_pointing(ms_name="invalid_name.ms")
#
#     except Exception:
#         return
#
#     assert False
