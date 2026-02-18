import os
import shutil
import pytest
import pathlib

import toolviper

from astrohack import open_locit
from astrohack.extract_locit import extract_locit
from astrohack.utils.verification_tools import add_data_folder_to_names_in_class


class TestExtractLocit:
    data_dir = "extract_locit_data"
    cal_table_name = "locit-input-pha.cal"

    def_lct_name = "locit-input-pha.locit.zarr"
    ref_lct_name = "locit-input-pha-reference.locit.zarr"

    ant_id = "ea17"
    ant_key = f"ant_{ant_id}"
    ddi_id = 0
    ddi_key = f"ddi_{ddi_id}"

    @classmethod
    def setup_class(cls):
        """
        Setup any state specific to the execution of the given test class
        such as fetching test data
        """

        toolviper.utils.data.download(file=cls.cal_table_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_lct_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        # shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """
        Create locit file with a cal-table and check it is created correctly.
        """

        # Create locit_mds and check the dictionary structure
        new_lct_mds = extract_locit(cal_table=self.cal_table_name, overwrite=True)
        assert pathlib.Path(
            self.def_lct_name
        ).is_dir(), f"A .locit.zarr file named {self.def_lct_name} does not exist."

        ref_lct_mds = open_locit(self.ref_lct_name)
        assert new_lct_mds.is_close_to(
            ref_lct_mds
        ), "Reference and new mdses are different."

    @pytest.mark.skip(reason="Fix later")
    def test_data_selection(self):
        """
        Check that only specified antenna is processed.
        """

        locit_mds = extract_locit(
            cal_table=self.cal_table, locit_name=self.locit_name, ant="ea17"
        )

        # There should only be 1 antenna in the dict named ea17
        assert len(locit_mds.keys()) == 1

        # Check that only the specific antenna is in the keys.
        assert list(locit_mds.keys()) == [
            "ant_ea17",
        ]

    @pytest.mark.skip(reason="Fix later")
    def test_extract_locit_ddi(self):
        """
        Check that only specified ddi is processed.
        """

        locit_mds = extract_locit(
            cal_table=self.cal_table, locit_name=self.locit_name, ddi=0
        )

        # Check that only the specific ddi is in the keys.
        assert len(locit_mds["ant_ea01"].keys()) == 1
        assert list(locit_mds["ant_ea01"].keys()) == ["ddi_0"]

    @pytest.mark.skip(reason="Fix later")
    def test_extract_locit_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """

        extract_locit(cal_table=self.cal_table, locit_name=self.locit_name)

        # To check this properly we need to not only know an exception was not thrown but that the file is ACTUALLY
        # overwritten. We do this by checking the modification time.
        initial_time = os.path.getctime(self.locit_name)

        extract_locit(
            cal_table=self.cal_table, locit_name=self.locit_name, overwrite=True
        )

        modified_time = os.path.getctime(self.locit_name)

        assert initial_time != modified_time

    @pytest.mark.skip(reason="Fix later")
    def test_extract_locit_no_overwrite(self):
        """
        Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
        extract_locit(cal_table=self.cal_table, locit_name=self.locit_name)

        initial_time = os.path.getctime(self.locit_name)

        try:
            extract_locit(
                cal_table=self.cal_table, locit_name=self.locit_name, overwrite=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime(self.locit_name)

            assert initial_time == modified_time
