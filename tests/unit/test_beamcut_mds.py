import pytest
import shutil

from toolviper.utils import data

from astrohack import open_beamcut


class TestBeamcut:
    data_folder = "beamcut_data"
    destination_folder = "beamcut_exports"

    ms_name = "kband_beamcut_small.ms"
    point_name = "kband_beamcut_small.point.zarr"
    holog_name = "kband_beamcut_small.holog.zarr"
    local_beamcut_name = "kband_beamcut_small_local.beamcut.zarr"
    remote_beamcut_name = "kband_beamcut_small.beamcut.zarr"
    ea15_report = "beamcut_exports/beamcut_report_ant_ea15_ddi_0.txt"

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
        shutil.rmtree(cls.data_folder, ignore_errors=True)
        shutil.rmtree(cls.destination_folder, ignore_errors=True)

    def test_init_beamcut(self):
        return

    def test_open_beamcut(self):
        return

    def test_beamcut_setitem(self):
        return

    def test_beamcut_getitem(self):
        return

    def test_beamcut_open(self):
        # .is_open to be tested here as well
        return

    def test_beamcut_keys(self):
        return

    def test_beamcut_summary(self):
        return

    def test_beamcut_observation_summary(self):
        return

    def test_beamcut_plots(self):
        # plot_beamcut_in_amplitude
        # plot_beamcut_in_attenuation
        # plot_beam_cuts_over_sky

        return

    def test_beam_fit_report(self):
        return
