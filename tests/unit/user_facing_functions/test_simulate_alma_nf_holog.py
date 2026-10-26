import shutil

import numpy as np
import pytest

from astrohack import estimate_nf_distance, open_holog, simulate_alma_nf_holog
from astrohack.core.estimate_nf_distance import _fit_minimum, _residual_surface_rms
from astrohack.holog import holog
from astrohack.utils.verification_tools import execute_cleanup

TF04_DISTANCE, TF04_FOCUS_OFFSET = 278.6315, 0.0988
# A small raster and coarse aperture sampling keep the tests fast
SMALL = {"map_size": 0.008, "map_cell": 2.0e-4, "aperture_cell": 0.15}


class TestSimulateAlmaNfHolog:
    holog_name = "sim_nf.holog.zarr"
    template_name = "sim_nf_template.holog.zarr"
    image_name = "sim_nf.image.zarr"

    @classmethod
    def setup_class(cls):
        cls.holog_mds = simulate_alma_nf_holog(
            holog_name=cls.holog_name, overwrite=True, **SMALL
        )

    @classmethod
    def teardown_class(cls):
        if execute_cleanup():
            for name in (cls.holog_name, cls.template_name, cls.image_name):
                shutil.rmtree(name, ignore_errors=True)

    def test_layout(self):
        """The file opens with open_holog and has the layout of extract_alma_nf_holog's."""
        holog_mds = open_holog(self.holog_name)
        xds = holog_mds["ant_DV08"]["ddi_0"]["map_0"].dataset
        n_samp = 41 * 41
        assert xds.VIS.shape == (n_samp, 1, 2)
        assert xds.DIRECTIONAL_COSINES.shape == (n_samp, 2)
        assert xds.pol.values.tolist() == ["I", "R2"]
        assert xds.attrs["near_field"]
        assert xds.attrs["summary"]["general"]["station"] == "TF04"
        assert xds.attrs["simulation"]["distance"] == TF04_DISTANCE
        assert xds.attrs["simulation"]["focus_offset"] == TF04_FOCUS_OFFSET
        assert np.abs(xds.VIS.values[:, 0, 0]).max() == pytest.approx(1.0)

    def test_template(self):
        """A template holog file gives its directions, times and frequency to the simulation."""
        template_mds = simulate_alma_nf_holog(
            holog_name=self.template_name,
            template_holog_name=self.holog_name,
            distance=300.0,
            overwrite=True,
            aperture_cell=0.15,
        )
        new = template_mds["ant_DV08"]["ddi_0"]["map_0"].dataset
        old = open_holog(self.holog_name)["ant_DV08"]["ddi_0"]["map_0"].dataset
        np.testing.assert_array_equal(
            new.DIRECTIONAL_COSINES.values, old.DIRECTIONAL_COSINES.values
        )
        np.testing.assert_array_equal(new.time.values, old.time.values)
        assert new.attrs["simulation"]["distance"] == 300.0

    def test_distance_override(self):
        """holog's nf_distance changes the near field correction: the residual grows away from the true distance."""
        residuals = []
        for distance in (TF04_DISTANCE, TF04_DISTANCE + 20.0):
            holog(
                holog_name=self.holog_name,
                image_name=self.image_name,
                nf_distance=distance,
                to_stokes=False,
                overwrite=True,
            )
            residuals.append(_residual_surface_rms(self.image_name, 1.0, 5.5))
        assert residuals[1] > 2 * residuals[0]

    def test_estimate_nf_distance(self):
        """The distance of the simulated transmitter is recovered from the data."""
        result = estimate_nf_distance(
            holog_name=self.holog_name,
            distances=[270.0, 274.0, 278.0, 282.0, 286.0],
        )
        assert result["distance"] == pytest.approx(TF04_DISTANCE, abs=2.0)
        assert len(result["residuals"]) == 5
        assert result["residual"] <= min(result["residuals"])


def test_fit_minimum():
    """The squared residual parabola gives the minimum; diverged trials are left out."""
    distances = np.array([270.0, 274.0, 278.0, 282.0, 286.0, 290.0])
    residuals = np.sqrt(4.0 + 0.25 * (distances - 277.0) ** 2)
    residuals[-1] = 300.0
    best, best_residual, converged = _fit_minimum(distances, residuals, 3.0)
    assert best == pytest.approx(277.0)
    assert best_residual == pytest.approx(2.0)
    assert converged.tolist() == [True, True, True, True, True, False]


def test_fit_minimum_without_parabola():
    """With fewer than three converged trials the smallest residual is used."""
    distances = np.array([270.0, 280.0, 290.0])
    residuals = np.array([300.0, 5.0, 400.0])
    best, best_residual, converged = _fit_minimum(distances, residuals, 3.0)
    assert (best, best_residual) == (280.0, 5.0)
    assert converged.tolist() == [False, True, False]
