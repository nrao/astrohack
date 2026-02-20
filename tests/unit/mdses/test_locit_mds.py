import shutil
import matplotlib

from toolviper.utils import data

from astrohack import AstrohackLocitFile, open_locit
from astrohack.utils.verification_tools import (
    are_lists_equal,
    are_png_files_close,
    is_captured_output_equal_to_txt_reference,
    add_data_folder_to_names_in_class,
)

matplotlib.use("Agg")


class TestLocitMDS:
    data_dir = "locit_data"
    destination_folder = "locit_exports"
    ref_products_name = f"ref_locit_products"

    locit_name = "locit-input-pha-reference.locit.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.locit_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_locit_mds_init(self):
        locit_mds = AstrohackLocitFile(self.locit_name)
        assert isinstance(locit_mds, AstrohackLocitFile)

    def test_locit_mds_text_exports(self):
        locit_mds = open_locit(self.locit_name)

        assert is_captured_output_equal_to_txt_reference(
            locit_mds.print_source_table,
            f"{self.ref_products_name}/src_tab_reference.txt",
        ), "Source table should be exactly equal to reference source table"

        assert is_captured_output_equal_to_txt_reference(
            locit_mds.print_array_configuration,
            f"{self.ref_products_name}/array_cfg_reference.txt",
        ), "Array configuration should be exactly equal to reference array configuration"

    def test_locit_mds_plot_exports(self):
        locit_mds = open_locit(self.locit_name)

        src_fk5_plot_name = "locit_source_table_fk5.png"
        locit_mds.plot_source_positions(self.destination_folder, precessed=False)
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{src_fk5_plot_name}",
            f"{self.ref_products_name}/{src_fk5_plot_name}",
        )
        assert (
            equal
        ), f"{msg}: FK5 source position plot should be exactly equal to reference FK5 source position plot"

        src_prece_plot_name = "locit_source_table_precessed.png"
        locit_mds.plot_source_positions(self.destination_folder, precessed=True)
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{src_prece_plot_name}",
            f"{self.ref_products_name}/{src_prece_plot_name}",
        )
        assert (
            equal
        ), f"{msg}: Precessed source position plot should be exactly equal to reference precessed source position plot"

        array_cfg_plot_name = "locit_array_configuration.png"
        locit_mds.plot_array_configuration(self.destination_folder)
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{array_cfg_plot_name}",
            f"{self.ref_products_name}/{array_cfg_plot_name}",
        )
        assert (
            equal
        ), f"{msg}: Array configuration plot should be exactly equal to reference array configuration plot"

    def test_locit_mds_metadata_style(self):
        locit_mds = open_locit(self.locit_name)

        assert "source_dict" in list(
            locit_mds.root.attrs.keys()
        ), "Root attributes should contain 'source_dict'"

        expected_src_keys = ["fk5", "id", "name", "precessed"]
        src_table = locit_mds.root.attrs["source_dict"]
        for key, value in src_table.items():
            assert key.isdigit(), "Source key should be a digit referencing field Ids"
            assert are_lists_equal(
                list(value.keys()), expected_src_keys
            ), "Source position keys should be the same as expected keys"

        expected_ant_keys = [
            "geocentric_position",
            "id",
            "latitude",
            "longitude",
            "name",
            "offset",
            "radius",
            "reference",
            "station",
        ]
        for ant_xdtree in locit_mds.values():
            assert "antenna_info" in list(
                ant_xdtree.attrs.keys()
            ), "Each antenna xarray DataTree needs to contain antenna info"
            antenna_info = ant_xdtree.attrs["antenna_info"]
            assert are_lists_equal(list(antenna_info.keys()), expected_ant_keys)
