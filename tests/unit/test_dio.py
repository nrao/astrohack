import shutil

import toolviper

from astrohack import holog, open_locit, open_position, open_beamcut
from astrohack.dio import open_holog
from astrohack.dio import open_image
from astrohack.dio import open_panel
from astrohack.dio import open_pointing

from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack import locit, extract_locit
from astrohack.panel import panel
from astrohack.beamcut import beamcut


class TestAstrohackDio:
    datafolder = "dioData"

    holography_ms_name = "ea25_cal_small_before_fixed.split.ms"
    point_name = "ea25_cal_small_before_fixed.split.point.zarr"
    holog_name = "ea25_cal_small_before_fixed.split.holog.zarr"
    image_name = "ea25_cal_small_before_fixed.split.image.zarr"
    panel_name = "ea25_cal_small_before_fixed.split.panel.zarr"

    holog_mds = None
    image_mds = None
    panel_mds = None

    locit_cal_table_name = "locit-input-pha.cal"
    locit_name = "locit-input-pha.locit.zarr"
    position_name = "locit-input-pha.position.zarr"

    locit_mds = None
    position_mds = None

    beamcut_ms_name = "kband_beamcut_small.ms"
    beamcut_point_name = "kband_beamcut_small.point.zarr"
    beamcut_holog_name = "kband_beamcut_small.holog.zarr"
    beamcut_name = "kband_beamcut_small.beamcut.zarr"

    beamcut_mds = None

    @classmethod
    def setup_class(cls):
        # Download all datasets to datafolder
        toolviper.utils.data.download(
            file=cls.holography_ms_name, folder=cls.datafolder
        )
        toolviper.utils.data.download(
            file=cls.locit_cal_table_name, folder=cls.datafolder
        )
        toolviper.utils.data.download(file=cls.beamcut_ms_name, folder=cls.datafolder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.datafolder}/{varvalue}")

        # Holography pre-processing
        extract_pointing(
            ms_name=cls.holography_ms_name,
            point_name=cls.point_name,
            parallel=True,
            overwrite=True,
        )

        cls.holog_mds = extract_holog(
            ms_name=cls.holography_ms_name,
            point_name=cls.point_name,
            holog_name=cls.holog_name,
            data_column="CORRECTED_DATA",
            parallel=True,
            overwrite=True,
        )

        cls.image_mds = holog(
            holog_name=cls.holog_name,
            image_name=cls.image_name,
            overwrite=True,
            phase_fit_engine="perturbations",
            to_stokes=True,
            parallel=True,
        )

        cls.panel_mds = panel(
            image_name=cls.image_name,
            panel_name=cls.panel_name,
            panel_model="rigid",
            parallel=True,
            overwrite=True,
        )

        # Antenna position corrections preprocessing
        cls.locit_mds = extract_locit(
            cal_table=cls.locit_cal_table_name,
            locit_name=cls.locit_name,
            overwrite=True,
        )

        cls.position_mds = locit(
            locit_name=cls.locit_name,
            position_name=cls.position_name,
            elevation_limit=10.0,
            polarization="both",
            fit_engine="scipy",
            parallel=False,
            overwrite=True,
        )

        # Beam cut preprocessing
        toolviper.utils.data.download(
            file="kband_beamcut_small.ms", folder=cls.datafolder
        )

        extract_pointing(
            ms_name=cls.beamcut_ms_name,
            point_name=cls.beamcut_point_name,
            overwrite=True,
        )

        extract_holog(
            ms_name=cls.beamcut_ms_name,
            point_name=cls.beamcut_point_name,
            holog_name=cls.beamcut_holog_name,
            data_column="DATA",
            overwrite=True,
        )

        cls.beamcut_mds = beamcut(
            holog_name=cls.beamcut_holog_name,
            beamcut_name=cls.beamcut_name,
            overwrite=True,
            destination=None,  # We don't want any products being saved to disk on this execution
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datafolder)

    def test_open_holog(self):
        """Open a holog file and return a holog data object"""
        holog_data = open_holog(self.holog_name)

        assert holog_data == self.holog_mds

    def test_open_image(self):
        """Open an image file and return an image data object"""
        image_data = open_image(self.holog_name)

        assert image_data == self.image_mds

    def test_open_panel(self):
        """Open a panel file and return a panel data object"""
        panel_data = open_panel(self.panel_name)

        assert panel_data == self.panel_mds

    def test_open_pointing(self):
        """Open a pointing file and return a pointing data object"""
        pointing_data = open_pointing(self.point_name)
        # check if keys match expected?
        # How to check xarray content...

        expected_keys = ["point_meta_ds", "ant_ea25", "ant_ea04", "ant_ea06"]

        for key in pointing_data.keys():
            assert key in expected_keys

    def test_open_locit(self):
        locit_data = open_locit(self.locit_name)

        assert locit_data == self.locit_mds

    def test_open_position(self):
        locit_data = open_position(self.position_name)

        assert locit_data == self.position_mds

    def test_open_beamcut(self):
        beamcut_data = open_beamcut(self.beamcut_name)

        assert beamcut_data == self.beamcut_mds
