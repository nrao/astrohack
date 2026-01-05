import pytest


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

        cls.beamcut_mds = open_beamcut(cls.remote_beamcut_name)
