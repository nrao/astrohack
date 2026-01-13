import pytest
import shutil
import os

from toolviper.utils import data

from astrohack import open_beamcut, AstrohackBeamcutFile
from astrohack.utils.validation import are_png_files_equal, capture_prints_from_function


class TestBeamcutMDS:
    data_folder = "beamcut_data"
    destination_folder = "beamcut_exports"
    ref_products_folder = f"{data_folder}/ref_beamcut_products"

    silly_name = "Anything"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.remote_beamcut_name, folder=cls.data_folder)
        data.download(file="ref_beamcut_products", folder=cls.data_folder)

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
        shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_beamcut_mds_init(self):
        beamcut_mds = AstrohackBeamcutFile(self.remote_beamcut_name)

        assert isinstance(beamcut_mds, AstrohackBeamcutFile)

    def test_beamcut_mds_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)
        summary_reference_name = f"{self.ref_products_folder}/summary_reference.txt"

        captured_output = capture_prints_from_function(beamcut_mds.summary)

        with open(summary_reference_name, "r") as ref_file:
            ref_content = ref_file.read()

        assert (
            captured_output == ref_content
        ), "Summary should be exactly equal to reference summary"

        return

    def test_beamcut_mds_observation_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        obs_summary_reference_name = (
            f"{self.ref_products_folder}/obs_summary_reference.txt"
        )
        local_obs_summary = f"{self.destination_folder}/obs_summary.txt"

        os.makedirs(self.destination_folder, exist_ok=True)
        beamcut_mds.observation_summary(local_obs_summary)

        with open(local_obs_summary, "r") as sum_file:
            local_obs_sum = sum_file.read()

        with open(obs_summary_reference_name, "r") as ref_file:
            ref_content = ref_file.read()

        assert (
            local_obs_sum == ref_content
        ), "Observation summary should be exactly equal to reference observation summary"
        return

    def test_beamcut_mds_plots(self):
        ant = "ea15"
        ddi = 0
        amp_plot_name = f"beamcut_amplitude_ant_{ant}_ddi_{ddi}.png"
        att_plot_name = f"beamcut_attenuation_ant_{ant}_ddi_{ddi}.png"
        lm_plot_name = f"beamcut_lm_offsets_ant_{ant}_ddi_{ddi}.png"

        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        beamcut_mds.plot_beamcut_in_amplitude(self.destination_folder, ant=ant, ddi=ddi)
        assert are_png_files_equal(
            f"{self.destination_folder}/{amp_plot_name}",
            f"{self.ref_products_folder}/{amp_plot_name}",
        ), "Amplitude plot png file is different from the expected png file"

        beamcut_mds.plot_beamcut_in_attenuation(
            self.destination_folder, ant=ant, ddi=ddi
        )
        assert are_png_files_equal(
            f"{self.destination_folder}/{att_plot_name}",
            f"{self.ref_products_folder}/{att_plot_name}",
        ), "Attenuation plot png file is different from the expected png file"

        beamcut_mds.plot_beam_cuts_over_sky(self.destination_folder, ant=ant, ddi=ddi)
        assert are_png_files_equal(
            f"{self.destination_folder}/{lm_plot_name}",
            f"{self.ref_products_folder}/{lm_plot_name}",
        ), "lm plot png file is different from the expected png file"

        return

    def test_beamcut_mds_fit_report(self):
        ant = "ea15"
        ddi = 0
        report_name = f"beamcut_report_ant_{ant}_ddi_{ddi}.txt"

        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        beamcut_mds.create_beam_fit_report(self.destination_folder, ant=ant, ddi=ddi)

        with open(f"{self.destination_folder}/{report_name}", "r") as local_report_file:
            local_rep = local_report_file.read()

        with open(
            f"{self.ref_products_folder}/{report_name}", "r"
        ) as remote_report_file:
            ref_rep = remote_report_file.read()

        assert local_rep == ref_rep, "Local and reference beamfit reports do not match"
        return
