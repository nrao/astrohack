import os
import pathlib
import shutil
import pytest
import toolviper

from astrohack import open_pointing
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
        toolviper.utils.data.download(file=cls.ref_pnt_name, folder=cls.data_dir)
        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a setup_class."""
        shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """Test extract_pointing with default parameters"""

        new_pnt_mds = extract_pointing(ms_name=self.ms_name, overwrite=True)
        assert pathlib.Path(
            self.def_pnt_name
        ).is_dir(), f"A .point.zarr file named {self.def_pnt_name} does not exist."

        ref_pnt_mds = open_pointing(self.ref_pnt_name)
        assert new_pnt_mds == ref_pnt_mds, "Reference and new mdses are different."

    def test_renaming(self):
        """Test extract_pointing naming"""
        new_pnt_mds = extract_pointing(
            ms_name=self.ms_name, point_name=self.alt_pnt_name, overwrite=True
        )

        assert pathlib.Path(
            self.alt_pnt_name
        ).is_dir(), f"A .point.zarr file named {self.alt_pnt_name} does not exist."

        assert (
            new_pnt_mds.filename == self.alt_pnt_name
        ), "Point mds filename does not match the file on disk."

        ref_pnt_mds = open_pointing(self.ref_pnt_name)
        assert new_pnt_mds == ref_pnt_mds, "Reference and new mdses are different."

    def test_antenna_exclusion(self):
        """Test extract_pointing antenna exclusion"""
        excluded_antenna = "ea06"
        new_pnt_mds = extract_pointing(
            ms_name=self.ms_name, exclude=excluded_antenna, overwrite=True
        )

        assert (
            f"ant_{excluded_antenna}" not in new_pnt_mds.keys()
        ), "Excluded antenna should not be present in the new mds."

    def test_invalid_ms_name(self):
        """Test extract_pointing with invalid ms name"""
        bogus_ms = "non-existe.ms"
        with pytest.raises(FileNotFoundError):
            extract_pointing(ms_name=bogus_ms, overwrite=True)

    def test_overwrite(self):
        """Test extract_pointing overwrite behaviour"""
        with pytest.raises(FileExistsError):
            extract_pointing(
                ms_name=self.ms_name, point_name=self.ref_pnt_name, overwrite=False
            )
