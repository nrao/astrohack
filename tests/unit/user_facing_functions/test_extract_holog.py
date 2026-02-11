import os
import json
import pathlib
import shutil

import pytest
import toolviper

from astrohack import open_holog
from astrohack.core.holog_obs_dict import HologObsDict
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict
from astrohack.utils.verification_tools import add_data_folder_to_names_in_class


class TestExtractHolog:
    data_dir = "ext_holog_data"
    ms_name = "ea25_cal_small_before_fixed.split.ms"
    pnt_name = "ea25_cal_small_before_reference.point.zarr"

    def_hlg_name = "ea25_cal_small_before_fixed.split.holog.zarr"
    ref_hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(file=cls.ms_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.pnt_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_hlg_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        # shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        new_hlg_mds = extract_holog(
            ms_name=self.ms_name, point_name=self.pnt_name, overwrite=True
        )
        assert pathlib.Path(
            self.def_hlg_name
        ).is_dir(), f"A .holog.zarr file named {self.def_hlg_name} does not exist."

        ref_hlg_mds = open_holog(self.ref_hlg_name)
        assert new_hlg_mds == ref_hlg_mds, "Reference and new mdses are different."

    def test_holog_obs_dict(self):
        """
        Specify a holography observations dictionary and check that the proper dictionary is created.
        """
        # Generate a holog observations dictionary with a subset of data described by ddi=1
        loc_hlg_obs_dict = generate_holog_obs_dict(
            point_name=self.pnt_name,
            baseline_average_distance="all",
            baseline_average_nearest="all",
            parallel=False,
        )

        loc_hlg_obs_dict.select(key="ddi", selected_values="0")

        # Extract holography data using holog_obd_dict
        extract_holog(
            ms_name=self.ms_name,
            point_name=self.pnt_name,
            holog_name=self.def_hlg_name,
            holog_obs_dict=loc_hlg_obs_dict,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        hlg_obs_dict_from_file = HologObsDict.from_holog_file(self.def_hlg_name)
        assert (
            hlg_obs_dict_from_file == loc_hlg_obs_dict
        ), "holog obs dict stored in the holog file is not the same as the one given as input"

    def test_ddi_and_ant_selection(self):
        """
        Specify a ddi value to be process and check that it is the only one processed.
        """
        ddi_id = 0
        ddi_key = f"ddi_{ddi_id}"
        ant_name = "ea25"
        ant_key = f"ant_{ant_name}"

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name=self.ms_name,
            point_name=self.pnt_name,
            holog_name=self.def_hlg_name,
            ddi=ddi_id,
            ant=ant_name,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        ant_list = list(holog_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert ant_list[0] == ant_key, "Ant name should be the same as the one given."

        ddi_list = list(holog_mds[ant_key].keys())
        assert len(ddi_list) == 1, "A single ddi should be present."
        assert ddi_list[0] == ddi_key, "DDI key should be the same as the one given."

    def test_overwrite(self):
        """
        Specify that the output file should be overwritten if it exists; check that it is overwritten.
        """
        initial_time = os.path.getctime(self.def_hlg_name)

        extract_holog(
            ms_name=self.ms_name,
            point_name=self.pnt_name,
            holog_name=self.def_hlg_name,
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )
        final_time = os.path.getctime(self.def_hlg_name)

        assert (
            initial_time != final_time
        ), "Recreated files has to have a different time from the original file."

        with pytest.raises(FileExistsError):
            extract_holog(
                ms_name=self.ms_name,
                point_name=self.pnt_name,
                holog_name=self.def_hlg_name,
                data_column="CORRECTED_DATA",
                parallel=False,
                overwrite=False,
            )

    def test_baseline_average_selection(self):
        """
        Run extract_holog using the baseline average distance as a filter; check that only the baselines with this
        average distance are returned.
        """
        # Extract holography data
        holog_mds = extract_holog(
            ms_name=self.ms_name,
            point_name=self.pnt_name,
            baseline_average_distance=195.1,
            baseline_average_nearest="all",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )
        # Check that the expected antenna is present.
        assert list(holog_mds.keys()) == [
            "ant_ea25"
        ], "After baseline distance selection, the holog_mds should contain holography data for only ea25"

        with pytest.raises(RuntimeError):
            extract_holog(
                ms_name=self.ms_name,
                point_name=self.pnt_name,
                baseline_average_nearest=1,
                baseline_average_distance=195.1,
                data_column="CORRECTED_DATA",
                parallel=False,
                overwrite=True,
            )
