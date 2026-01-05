import pytest
import shutil
import os
import io
import contextlib
import xarray
import hashlib

from collections.abc import KeysView

from toolviper.utils import data

from astrohack import open_beamcut, AstrohackBeamcutFile


def are_binary_files_equal(file_a, file_b):
    with open(file_a, "rb") as bin_file_a:
        hash_a = hashlib.md5(bin_file_a.read()).hexdigest()
    with open(file_b, "rb") as bin_file_b:
        hash_b = hashlib.md5(bin_file_b.read()).hexdigest()
    return hash_a == hash_b


class TestBeamcut:
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

    def test_init_and_open_beamcut(self):
        beamcut_mds = AstrohackBeamcutFile(self.silly_name)

        assert (
            beamcut_mds.file == self.silly_name
        ), "Beamcut mds file name should be the same as the one given as argument to __init__"

        assert not beamcut_mds.is_open, "Beamcut mds file should not be opened yet"

        sucessful_open = beamcut_mds.open()
        assert (
            not sucessful_open
        ), "opening beamcut_mds file should fail when beamcut_mds.file is set to nonsense"

        sucessful_open = beamcut_mds.open(self.remote_beamcut_name)
        assert (
            sucessful_open
        ), "Opening beamcut should succeed now that the correct file name is given"
        assert (
            beamcut_mds.is_open
        ), "is_open property needs to return True now that the file has been opened"
        assert (
            beamcut_mds.file == self.remote_beamcut_name
        ), ".file attribute should now be set to the name of the given file."

        return

    def test_beamcut_keys_getitem_and_setitem(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        old_xdt_keys = beamcut_mds.keys()
        assert isinstance(
            old_xdt_keys, KeysView
        ), "Keys method should return a dict_keys object"
        assert len(old_xdt_keys) == 2, "File should contain 2 antenna subtrees"

        ant_17_subtree = beamcut_mds["ant_ea17"]
        assert isinstance(ant_17_subtree, xarray.DataTree)

        beamcut_mds["ant_ea19"] = ant_17_subtree
        new_xdt_keys = beamcut_mds.keys()
        assert len(new_xdt_keys) == 3, "File should now contain 3 antenna subtrees"
        assert (
            "ant_ea19" in new_xdt_keys
        ), "New antenna subtree should appear amongst keys"

        return

    def test_beamcut_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)
        summary_reference_name = f"{self.ref_products_folder}/summary_reference.txt"

        output_capture = io.StringIO()

        # Use redirect_stdout to capture the function's output
        with contextlib.redirect_stdout(output_capture):
            beamcut_mds.summary()

        # Get the captured output as a string
        captured_output = output_capture.getvalue()

        with open(summary_reference_name, "r") as ref_file:
            ref_content = ref_file.read()
        assert (
            captured_output == ref_content
        ), "Summary should be exactly equal to reference summary"

        return

    def test_beamcut_observation_summary(self):
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

    def test_beamcut_plots(self):
        ant = "ea15"
        ddi = 0
        amp_plot_name = f"beamcut_amplitude_ant_{ant}_ddi_{ddi}.png"
        att_plot_name = f"beamcut_attenuation_ant_{ant}_ddi_{ddi}.png"
        lm_plot_name = f"beamcut_lm_offsets_ant_{ant}_ddi_{ddi}.png"

        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        beamcut_mds.plot_beamcut_in_amplitude(self.destination_folder, ant=ant, ddi=ddi)
        assert are_binary_files_equal(
            f"{self.destination_folder}/{amp_plot_name}",
            f"{self.ref_products_folder}/{amp_plot_name}",
        ), "Amplitude plot hash is different from the expected hash"

        beamcut_mds.plot_beamcut_in_attenuation(
            self.destination_folder, ant=ant, ddi=ddi
        )
        assert are_binary_files_equal(
            f"{self.destination_folder}/{att_plot_name}",
            f"{self.ref_products_folder}/{att_plot_name}",
        ), "Attenuation plot hash is different from the expected hash"

        beamcut_mds.plot_beam_cuts_over_sky(self.destination_folder, ant=ant, ddi=ddi)
        assert are_binary_files_equal(
            f"{self.destination_folder}/{lm_plot_name}",
            f"{self.ref_products_folder}/{lm_plot_name}",
        ), "lm plot hash is different from the expected hash"

        return

    def test_beam_fit_report(self):
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
