import shutil
import xarray
import pathlib

from collections.abc import KeysView, ValuesView, ItemsView

from toolviper.utils import data

import astrohack
from astrohack.io.base_mds import AstrohackBaseFile


class TestBaseMds:
    """Here we use a beamcut file as an example of a base file"""

    data_folder = "base_mds_data"

    silly_name = "Anything"
    beamcut_file_name = "kband_beamcut_small.beamcut.zarr"
    output_file_name = "base_mds_test.base.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.beamcut_file_name, folder=cls.data_folder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.data_folder}/{varvalue}")

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_folder, ignore_errors=True)
        return

    def test_init_and_open_base_mds(self):
        base_mds = AstrohackBaseFile(self.silly_name)

        assert (
            base_mds.filename == self.silly_name
        ), "base mds file name should be the same as the one given as argument to __init__"

        assert not base_mds.is_open, "base mds file should not be opened yet"

        sucessful_open = base_mds.open()
        assert (
            not sucessful_open
        ), "opening base_mds file should fail when base_mds.file is set to nonsense"

        sucessful_open = base_mds.open(self.beamcut_file_name)
        assert (
            sucessful_open
        ), "Opening base_mds should succeed now that the correct file name is given"
        assert (
            base_mds.is_open
        ), "is_open property needs to return True now that the file has been opened"
        assert (
            base_mds.filename == self.beamcut_file_name
        ), ".file attribute should now be set to the name of the given file."

        return

    def test_base_mds_keys_getitem_and_setitem(self):
        base_mds = AstrohackBaseFile(self.beamcut_file_name)
        base_mds.open()

        old_xdt_keys = base_mds.keys()
        assert isinstance(
            old_xdt_keys, KeysView
        ), "Keys method should return a dict_keys object"
        assert len(old_xdt_keys) == 2, "File should contain 2 antenna subtrees"

        ant_17_subtree = base_mds["ant_ea17"]
        assert isinstance(ant_17_subtree, xarray.DataTree)

        base_mds["ant_ea19"] = ant_17_subtree
        new_xdt_keys = base_mds.keys()
        assert len(new_xdt_keys) == 3, "File should now contain 3 antenna subtrees"
        assert (
            "ant_ea19" in new_xdt_keys
        ), "New antenna subtree should appear amongst keys"

        return

    def test_base_mds_write_items_and_values_and_create_from_input_parameters(self):
        base_mds = AstrohackBaseFile(self.beamcut_file_name)
        base_mds.open()

        old_xdt_values = base_mds.values()
        assert isinstance(
            old_xdt_values, ValuesView
        ), "Values method should return a ValuesView object"
        assert len(old_xdt_values) == 2, "File should contain 2 antenna subtrees"

        any_ant_subtree = list(old_xdt_values)[0]
        assert isinstance(any_ant_subtree, xarray.DataTree)

        old_xdt_items = base_mds.items()
        assert isinstance(
            old_xdt_items, ItemsView
        ), "Items method should return a ItemsView object"

        test_input_pars = {
            "adenine": "a",
            "guanine": "g",
            "thymine": "t",
            "cytosine": "c",
        }
        new_base_mds = AstrohackBaseFile.create_from_input_parameters(
            self.output_file_name, test_input_pars
        )

        assert isinstance(
            new_base_mds.root, xarray.DataTree
        ), "Root attribute must be a DataTree"

        assert (
            len(new_base_mds.root.children) == 0
        ), "Root data tree must not contain any children"

        assert (
            new_base_mds.root.attrs["input_parameters"] == test_input_pars
        ), "input_parameters should be equal to the given dictionary"

        ref_origin_dict = {
            "origin": "astrohack",
            "version": astrohack.__version__,
            "creator_function": "test_base_mds_write_items_and_values_and_create_from_input_parameters",
        }
        assert (
            new_base_mds.root.attrs["origin_info"] == ref_origin_dict
        ), "Origin information should be equal to reference"

        # Add a child
        new_base_mds["ant_any"] = any_ant_subtree
        new_base_mds.write()

        # Delete old objs for better testing
        del base_mds, new_base_mds

        assert pathlib.Path(
            self.output_file_name
        ).is_dir(), f"Write method should create a directory named {self.output_file_name} containiing the DataTree"

        new_base_mds = AstrohackBaseFile(self.output_file_name)
        new_base_mds.open()
        assert (
            new_base_mds.root.attrs["input_parameters"] == test_input_pars
        ), "input_parameters from file on disk should be equal to the given dictionary"

        assert (
            len(new_base_mds.keys()) == 1
        ), "Disk read from file should contain one single key"

        assert (
            new_base_mds["ant_any"] == any_ant_subtree
        ), "Antenna subtree in writen file should be equal to the one given to it"
