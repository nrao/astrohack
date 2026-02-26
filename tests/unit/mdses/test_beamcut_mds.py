import shutil
import os
import matplotlib

from toolviper.utils import data

from astrohack import open_beamcut, AstrohackBeamcutFile
from astrohack.utils.verification_tools import (
    are_png_files_close,
    are_txt_files_equal,
    add_data_folder_to_names_in_class,
)

matplotlib.use("Agg")


class TestBeamcutMDS:
    data_dir = "beamcut_data"
    destination_folder = "beamcut_exports"
    ref_products_name = f"ref_beamcut_products"

    silly_name = "Anything"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.remote_beamcut_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return

    def test_beamcut_mds_init(self):
        beamcut_mds = AstrohackBeamcutFile(self.remote_beamcut_name)

        assert isinstance(beamcut_mds, AstrohackBeamcutFile)

    def test_beamcut_mds_observation_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        obs_summary_reference_name = (
            f"{self.ref_products_name}/obs_summary_reference.txt"
        )
        local_obs_summary = f"{self.destination_folder}/obs_summary.txt"

        os.makedirs(self.destination_folder, exist_ok=True)
        beamcut_mds.observation_summary(local_obs_summary)

        assert are_txt_files_equal(
            local_obs_summary, obs_summary_reference_name
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
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{amp_plot_name}",
            f"{self.ref_products_name}/{amp_plot_name}",
        )
        assert (
            equal
        ), f"{msg}: Amplitude plot png file is different from the expected png file"

        beamcut_mds.plot_beamcut_in_attenuation(
            self.destination_folder, ant=ant, ddi=ddi
        )
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{att_plot_name}",
            f"{self.ref_products_name}/{att_plot_name}",
        )
        assert (
            equal
        ), f"{msg}: Attenuation plot png file is different from the expected png file"

        beamcut_mds.plot_beam_cuts_over_sky(self.destination_folder, ant=ant, ddi=ddi)
        equal, msg = are_png_files_close(
            f"{self.destination_folder}/{lm_plot_name}",
            f"{self.ref_products_name}/{lm_plot_name}",
        )
        assert equal, f"{msg}: lm plot png file is different from the expected png file"

        return

    def test_beamcut_mds_fit_report(self):
        ant = "ea15"
        ddi = 0
        report_name = f"beamcut_report_ant_{ant}_ddi_{ddi}.txt"

        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        beamcut_mds.create_beam_fit_report(self.destination_folder, ant=ant, ddi=ddi)

        with open(f"{self.destination_folder}/{report_name}", "r") as local_report_file:
            local_rep = local_report_file.read()

        with open(f"{self.ref_products_name}/{report_name}", "r") as remote_report_file:
            ref_rep = remote_report_file.read()

        assert local_rep == ref_rep, "Local and reference beamfit reports do not match"
        return
