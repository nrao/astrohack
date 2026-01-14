import os
import shutil

from toolviper.utils import data

from astrohack import AstrohackPositionFile, extract_locit, locit, open_position
from astrohack.utils.ray_tracing_general import return_line
from astrohack.utils.validation import (
    capture_prints_from_function,
    are_png_files_equal,
    are_lists_equal,
    are_txt_files_equal,
    is_captured_output_equal_to_txt_reference,
)


class TestPositionMDS:
    data_folder = "locit_data"
    destination_folder = "position_exports"
    ref_products_folder = f"{data_folder}/ref_position_products"

    phase_cal_table_name = "locit-input-pha.cal"
    locit_name = "ant-pos.locit.zarr"
    position_no_comb_name = "ant-pos-no-comb.position.zarr"
    position_simple_comb_name = "ant-pos-simple-comb.position.zarr"
    position_diff_comb_name = "ant-pos-diff-comb.position.zarr"

    position_files = {
        "no": position_no_comb_name,
        "simple": position_simple_comb_name,
        "difference": position_diff_comb_name,
    }

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.phase_cal_table_name, folder=cls.data_folder)
        # data.download(file="ref_position_products", folder=cls.data_folder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.data_folder}/{varvalue}")

        for key in cls.position_files.keys():
            cls.position_files[key] = f"{cls.data_folder}/{cls.position_files[key]}"

        extract_locit(cls.phase_cal_table_name, cls.locit_name, overwrite=True)
        locit(
            cls.locit_name, cls.position_no_comb_name, combine_ddis="no", overwrite=True
        )
        locit(
            cls.locit_name,
            cls.position_simple_comb_name,
            combine_ddis="simple",
            overwrite=True,
        )
        locit(
            cls.locit_name,
            cls.position_diff_comb_name,
            combine_ddis="difference",
            overwrite=True,
        )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        # shutil.rmtree(cls.data_folder, ignore_errors=True)
        # shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_position_mds_init(self):
        position_mds = AstrohackPositionFile(self.position_no_comb_name)
        assert isinstance(position_mds, AstrohackPositionFile)

    def test_position_mds_summary(self):
        for label, filename in self.position_files.items():
            position_mds = open_position(filename)
            summary_reference_name = (
                f"{self.ref_products_folder}/summary_{label}_comb_reference.txt"
            )
            assert is_captured_output_equal_to_txt_reference(
                position_mds.summary, summary_reference_name
            ), (
                f"{label.capitalize()} combination summary should be exactly equal to reference {label} combination "
                f"summary"
            )
        return

    def test_position_mds_text_exports(self):
        pos_res_name_dict = {
            "no": "position_separated_ddis_fit_results.txt",
            "simple": "position_combined_simple_fit_results.txt",
            "difference": "position_combined_difference_fit_results.txt",
        }
        for label, filename in self.position_files.items():
            position_mds = open_position(filename)

            fit_res_filename = pos_res_name_dict[label]
            position_mds.export_locit_fit_results(self.destination_folder)
            assert are_txt_files_equal(
                f"{self.destination_folder}/{fit_res_filename}",
                f"{self.ref_products_folder}/{fit_res_filename}",
            ), f"{fit_res_filename} differs from reference file."

            parminator_filename = f"parminator_{label}_combination.par"
            position_mds.export_results_to_parminator(
                f"{self.destination_folder}/parminator_filename",
                correction_threshold=0.001,
                ddi=0,  # DDI specified for the no comb case
            )
            assert are_txt_files_equal(
                f"{self.destination_folder}/{parminator_filename}",
                f"{self.ref_products_folder}/{parminator_filename}",
            ), f"{parminator_filename} differs from reference file."
        return

    def test_position_mds_plot_exports(self):
        ddi = 0
        ant = "ea16"
        ant_pos_name_dict = {
            "no": "position_corrections_separated_ddi_0.png",
            "simple": "position_corrections_combined_simple.png",
            "difference": "position_corrections_combined_difference.png",
        }
        delay_name_dict = {
            "no": "position_delays_ant_ea16_separated_ddi_0.png",
            "simple": "position_delays_ant_ea16_combined_simple.png",
            "difference": "position_delays_ant_ea16_combined_difference.png",
        }
        sky_coverage_name_dict = {
            "no": "position_sky_coverage_ant_ea16_ddi_0.png",
            "simple": "position_sky_coverage_ant_ea16.png",
            "difference": "position_sky_coverage_ant_ea16.png",
        }

        for label, filename in self.position_files.items():
            position_mds = open_position(filename)

            position_mds.plot_sky_coverage(self.destination_folder, ant=ant, ddi=ddi)

            position_mds.plot_delays(self.destination_folder, ant=ant, ddi=ddi)

            position_mds.plot_position_corrections(self.destination_folder, ddi=ddi)

        #     src_fk5_plot_name = "position_source_table_fk5.png"
        #     position_mds.plot_source_positions(self.destination_folder, precessed=False)
        #     assert are_png_files_equal(
        #         f"{self.destination_folder}/{src_fk5_plot_name}",
        #         f"{self.ref_products_folder}/{src_fk5_plot_name}",
        #     ), "FK5 source position plot should be exactly equal to reference FK5 source position plot"
        #
        #     src_prece_plot_name = "position_source_table_precessed.png"
        #     position_mds.plot_source_positions(self.destination_folder, precessed=True)
        #     assert are_png_files_equal(
        #         f"{self.destination_folder}/{src_prece_plot_name}",
        #         f"{self.ref_products_folder}/{src_prece_plot_name}",
        #     ), "Precessed source position plot should be exactly equal to reference precessed source position plot"
        #
        #     array_cfg_plot_name = "position_antenna_positions.png"
        #     position_mds.plot_array_configuration(self.destination_folder)
        #     assert are_png_files_equal(
        #         f"{self.destination_folder}/{array_cfg_plot_name}",
        #         f"{self.ref_products_folder}/{array_cfg_plot_name}",
        #     ), "Array configuration plot should be exactly equal to reference array configuration plot"
        return

    def test_position_mds_metadata_style(self):
        # position_mds = open_position(self.position_name)
        #
        # assert "source_dict" in list(
        #     position_mds.root.attrs.keys()
        # ), "Root attributes should contain 'source_dict'"
        #
        # expected_src_keys = ["fk5", "id", "name", "precessed"]
        # src_table = position_mds.root.attrs["source_dict"]
        # for key, value in src_table.items():
        #     assert key.isdigit(), "Source key should be a digit referencing field Ids"
        #     assert are_lists_equal(
        #         list(value.keys()), expected_src_keys
        #     ), "Source position keys should be the same as expected keys"
        #
        # expected_ant_keys = [
        #     "geocentric_position",
        #     "id",
        #     "latitude",
        #     "longitude",
        #     "name",
        #     "offset",
        #     "radius",
        #     "reference",
        #     "station",
        # ]
        # for ant_xdtree in position_mds.values():
        #     assert "antenna_info" in list(
        #         ant_xdtree.attrs.keys()
        #     ), "Each antenna xarray DataTree needs to contain antenna info"
        #     antenna_info = ant_xdtree.attrs["antenna_info"]
        #     assert are_lists_equal(list(antenna_info.keys()), expected_ant_keys)
        return
