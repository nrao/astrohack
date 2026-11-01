import numpy as np
import pytest

from astrohack.utils.algorithms import compute_stokes
from astrohack.utils.conversion import (
    convert_5d_grid_from_stokes,
    convert_5d_grid_to_stokes,
)

# I, Q, U and V, complex as in a beam or aperture
i, q, u, v = 1.0 + 0.1j, 0.1 - 0.05j, 0.2 + 0.3j, 0.05 - 0.02j
STOKES = np.array([i, q, u, v])
# casacore's StokesConverter with rescale=true, which takes the correlations at the level of I
CORRELATIONS = {
    ("RR", "RL", "LR", "LL"): np.array([i + v, q + 1j * u, q - 1j * u, i - v]),
    ("XX", "XY", "YX", "YY"): np.array([i + q, u + 1j * v, u - 1j * v, i - q]),
}
FEEDS = ["circular", "linear"]


@pytest.mark.parametrize("pol", CORRELATIONS, ids=FEEDS)
def test_convert_5d_grid_to_stokes(pol):
    """Test that casacore's correlations convert to its Stokes parameters"""
    grid = CORRELATIONS[pol].reshape(1, 1, 4, 1, 1)
    np.testing.assert_allclose(
        convert_5d_grid_to_stokes(grid, list(pol)).ravel(), STOKES
    )


@pytest.mark.parametrize("pol", CORRELATIONS, ids=FEEDS)
def test_convert_5d_grid_from_stokes(pol):
    """Test that the Stokes parameters convert to casacore's correlations"""
    grid = STOKES.reshape(1, 1, 4, 1, 1)
    np.testing.assert_allclose(
        convert_5d_grid_from_stokes(grid, ["I", "Q", "U", "V"], list(pol)).ravel(),
        CORRELATIONS[pol],
    )


@pytest.mark.parametrize("pol", CORRELATIONS, ids=FEEDS)
def test_compute_stokes(pol):
    """Test that compute_stokes gives the amplitude and phase of casacore's Stokes parameters"""
    amplitude, phase, _, _ = compute_stokes(
        CORRELATIONS[pol][np.newaxis], np.ones((1, 4)), list(pol)
    )
    np.testing.assert_allclose(amplitude[0], np.abs(STOKES))
    np.testing.assert_allclose(phase[0], np.angle(STOKES, deg=True))
