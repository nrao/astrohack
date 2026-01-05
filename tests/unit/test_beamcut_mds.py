import pytest
import shutil
import os
import io
import contextlib
import xarray

from collections.abc import KeysView

from toolviper.utils import data

from astrohack import open_beamcut, AstrohackBeamcutFile


class TestBeamcut:
    data_folder = "beamcut_data"
    destination_folder = "beamcut_exports"

    silly_name = "azurbanipal"
    # ms_name = "kband_beamcut_small.ms"
    # point_name = "kband_beamcut_small.point.zarr"
    # holog_name = "kband_beamcut_small.holog.zarr"
    # local_beamcut_name = "kband_beamcut_small_local.beamcut.zarr"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"
    summary_reference_name = "summary_reference.txt"
    obs_summary_reference_name = "obs_summary_reference.txt"

    local_obs_summary = f"{destination_folder}/obs_summary.txt"
    # ea15_report = "beamcut_exports/beamcut_report_ant_ea15_ddi_0.txt"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.remote_beamcut_name, folder=cls.data_folder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.data_folder}/{varvalue}")

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        # shutil.rmtree(cls.data_folder, ignore_errors=True)
        # shutil.rmtree(cls.destination_folder, ignore_errors=True)
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

        output_capture = io.StringIO()

        # Use redirect_stdout to capture the function's output
        with contextlib.redirect_stdout(output_capture):
            beamcut_mds.summary()

        # Get the captured output as a string
        captured_output = output_capture.getvalue()

        with open(self.summary_reference_name, "r") as ref_file:
            ref_content = ref_file.read()
        assert (
            captured_output == ref_content
        ), "Summary should be exactly equal to reference summary"

        return

    def test_beamcut_observation_summary(self):
        beamcut_mds = open_beamcut(self.remote_beamcut_name)

        os.makedirs(self.destination_folder, exist_ok=True)
        beamcut_mds.observation_summary(self.local_obs_summary)

        with open(self.local_obs_summary, "r") as sum_file:
            local_obs_sum = sum_file.read()

        with open(self.obs_summary_reference_name, "r") as ref_file:
            ref_content = ref_file.read()

        assert (
            local_obs_sum == ref_content
        ), "Observation summary should be exactly equal to reference observation summary"
        return

    def test_beamcut_plots(self):
        # plot_beamcut_in_amplitude
        # plot_beamcut_in_attenuation
        # plot_beam_cuts_over_sky

        return

    def test_beam_fit_report(self):
        return
