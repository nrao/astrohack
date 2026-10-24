import pathlib
import shutil

import numpy as np
import pytest
import toolviper
import zarr

from astrohack import open_holog
from astrohack.extract_alma_nf_holog import extract_alma_nf_holog
from astrohack.holog import holog
from astrohack.utils.constants import clight
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
)


class TestExtractAlmaNfHolog:
    data_dir = "alma_nf_data"
    asdm_name = "uid___A002_Xad52b4_X3"
    def_holog_name = "uid___A002_Xad52b4_X3.holog.zarr"
    alt_holog_name = "nf_x3.holog.zarr"
    image_name = "nf_x3.image.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class"""
        toolviper.utils.data.download(file=cls.asdm_name, folder=cls.data_dir)
        add_data_folder_to_names_in_class(cls)
        cls.holog_mds = extract_alma_nf_holog(asdm_name=cls.asdm_name, overwrite=True)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a setup_class."""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """Test extract_alma_nf_holog with default parameters against the original filler's results; unlike the
        filler's arctan, the phase covers all four quadrants"""
        assert pathlib.Path(self.def_holog_name).is_dir(), (
            f"A .holog.zarr file named {self.def_holog_name} does not exist."
        )
        assert list(self.holog_mds.keys()) == ["ant_DV08"]

        map_xds = self.holog_mds["ant_DV08"]["ddi_0"]["map_0"].dataset
        assert map_xds.attrs["near_field"]
        assert dict(map_xds.sizes) == {"time": 55799, "chan": 1, "pol": 2, "lm": 2}
        assert map_xds.pol.values.tolist() == ["I", "R2"]
        assert map_xds.attrs["time_smoothing_interval"] == pytest.approx(
            0.04800001624971628
        )
        assert abs(map_xds.VIS.values[0, 0, 0]) == pytest.approx(0.0026938736)
        assert np.any(np.abs(np.angle(map_xds.VIS.values[:, 0, 0])) > np.pi / 2)
        assert map_xds.WEIGHT.values[:2, 0, 0].tolist() == [5.0, 4.0]
        assert map_xds.WEIGHT.values[-1, 0, 0] == 7.0

        beam = map_xds.attrs["summary"]["beam"]
        assert beam["l extent"] == pytest.approx(
            [-0.010765705250140348, 0.010765707429685939]
        )
        assert beam["m extent"] == pytest.approx(
            [-0.010751019298791298, 0.0108383490450486]
        )
        cell_size = clight / map_xds.chan.values[0] / 12.0 / 3.0
        n_pix = int(np.ceil((beam["l extent"][1] - beam["l extent"][0]) / cell_size))
        assert beam["cell size"] == pytest.approx(cell_size)
        assert beam["grid size"] == [n_pix, n_pix]

    def test_open_holog(self):
        """Test that open_holog accepts files created by extract_alma_nf_holog"""
        holog_mds = open_holog(self.def_holog_name)
        map_key = "ant_DV08/ddi_0/map_0"
        assert holog_mds[map_key].dataset.identical(self.holog_mds[map_key].dataset)

    def test_options(self):
        """Test extract_alma_nf_holog calibration, output and zarr format options"""
        holog_mds = extract_alma_nf_holog(
            asdm_name=self.asdm_name,
            holog_name=self.alt_holog_name,
            phase_calibration="none",
            plot_calibration=True,
            save_calibration=True,
            zarr_format=2,
            overwrite=True,
        )
        assert holog_mds.filename == self.alt_holog_name
        assert zarr.open_group(self.alt_holog_name, mode="r").metadata.zarr_format == 2

        base_name = self.alt_holog_name.removesuffix(".holog.zarr")
        cal_name = f"{base_name}.cal.zarr"
        assert zarr.open_group(cal_name, mode="r").metadata.zarr_format == 2
        assert pathlib.Path(f"{base_name}_amplitude_cal.png").is_file()
        assert not pathlib.Path(f"{base_name}_phase_cal.png").exists()

        new_vis = holog_mds["ant_DV08"]["ddi_0"]["map_0"].VIS.values
        def_vis = self.holog_mds["ant_DV08"]["ddi_0"]["map_0"].VIS.values
        assert new_vis.shape == def_vis.shape
        assert not np.allclose(new_vis[:, 0, 0], def_vis[:, 0, 0])

    def test_holog(self):
        """Test that holog produces near field apertures from the extracted data"""
        image_mds = holog(
            holog_name=self.def_holog_name,
            image_name=self.image_name,
            alma_osf_pad="TF03",
            to_stokes=False,
            overwrite=True,
        )
        image_xds = image_mds["ant_DV08"]["ddi_0"].dataset
        assert image_xds.pol.values.tolist() == ["I", "R2"]
        assert np.all(np.isfinite(image_xds.APERTURE.values))

        # The aperture is confined to the 12 m dish, which it is not with a folded phase
        amplitude = image_xds.AMPLITUDE.values[0, 0, 0]
        u_grid, v_grid = np.meshgrid(
            image_xds.u_prime.values, image_xds.v_prime.values, indexing="ij"
        )
        radius = np.hypot(u_grid, v_grid)
        outside = (radius > 6.5) & (radius < 8)
        assert np.nanmean(amplitude[radius < 6]) > 10 * np.nanmean(amplitude[outside])

    def test_invalid_asdm_name(self):
        """Test extract_alma_nf_holog with an invalid ASDM name"""
        with pytest.raises(FileNotFoundError):
            extract_alma_nf_holog(asdm_name="non-existent-asdm", overwrite=True)

    def test_overwrite(self):
        """Test extract_alma_nf_holog overwrite behaviour"""
        with pytest.raises(FileExistsError):
            extract_alma_nf_holog(asdm_name=self.asdm_name, overwrite=False)
