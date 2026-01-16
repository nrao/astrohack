import shutil

from toolviper.utils import data

from astrohack import AstrohackPositionFile, extract_locit, locit, open_position
from astrohack.utils.validation import (
    are_png_files_equal,
    are_txt_files_equal,
    is_captured_output_equal_to_txt_reference,
    are_png_files_equal_macos,
)


class TestPositionMDS:
    data_folder = "position_data"
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
        data.download(file="ref_position_products", folder=cls.data_folder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.data_folder}/{varvalue}")

        extract_locit(cls.phase_cal_table_name, cls.locit_name, overwrite=True)

        for key in cls.position_files.keys():
            cls.position_files[key] = f"{cls.data_folder}/{cls.position_files[key]}"
            locit(
                cls.locit_name,
                cls.position_files[key],
                combine_ddis=key,
                overwrite=True,
            )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_folder, ignore_errors=True)
        shutil.rmtree(cls.destination_folder, ignore_errors=True)
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
                f"{self.destination_folder}/{parminator_filename}",
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
            "simple": None,  # Simple and difference files are in thesis equal, but may differ in metadata, since they
            # have the same exact name, simple is set to None in order to avoid unreasonable test failures
            "difference": "position_sky_coverage_ant_ea16.png",
        }

        for label, filename in self.position_files.items():
            position_mds = open_position(filename)

            position_mds.plot_sky_coverage(self.destination_folder, ant=ant, ddi=ddi)
            if sky_coverage_name_dict[label] is not None:
                equal, msg = are_png_files_equal_macos(
                    f"{self.destination_folder}/{sky_coverage_name_dict[label]}",
                    f"{self.ref_products_folder}/{sky_coverage_name_dict[label]}",
                )
                assert (
                    equal
                ), f"{msg}: {sky_coverage_name_dict[label]} differs from reference file."

            position_mds.plot_delays(self.destination_folder, ant=ant, ddi=ddi)
            equal, msg = are_png_files_equal_macos(
                f"{self.destination_folder}/{delay_name_dict[label]}",
                f"{self.ref_products_folder}/{delay_name_dict[label]}",
            )
            assert (
                equal
            ), f"{msg}: {delay_name_dict[label]} differs from reference file."

            position_mds.plot_position_corrections(self.destination_folder, ddi=ddi)
            equal, msg = are_png_files_equal_macos(
                f"{self.destination_folder}/{ant_pos_name_dict[label]}",
                f"{self.ref_products_folder}/{ant_pos_name_dict[label]}",
            )
            assert (
                equal
            ), f"{msg}: {ant_pos_name_dict[label]} differs from reference file."

        return

    def test_position_mds_structure(self):
        depth_dict = {
            "no": 2,
            "simple": 1,
            "difference": 1,
        }

        for label, filename in self.position_files.items():
            position_mds = open_position(filename)
            expected_depth = depth_dict[label]
            assert (
                position_mds.root.depth == expected_depth
            ), f"{label.capitalize()} combination mds must have a depth of {expected_depth}."
            assert (
                "reference_antenna" in position_mds.root.attrs
            ), f"{label.capitalize()} combination mds must have a root attribute for reference antenna."
            assert (
                "telescope_name" in position_mds.root.attrs
            ), f"{label.capitalize()} combination mds must have a root attribute for telescopa name."
