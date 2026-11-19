import numpy as np
import pytest

from astrohack.utils.algorithms import calc_coords
from astrohack.utils.imaging import _compute_aperture_fft


class TestCalcCoords:
    @pytest.mark.parametrize("size", [64, 65])
    def test_fft_origin(self, size):
        """Test that a beam centred on l = m = 0 of the calc_coords axes transforms to an aperture without a phase
        tilt"""
        cell_size = np.array([-1e-4, 1e-4])
        l_axis, m_axis = calc_coords(np.array([size, size]), cell_size)
        assert l_axis[size // 2] == 0
        assert m_axis[size // 2] == 0

        l_grid, m_grid = np.meshgrid(l_axis, m_axis, indexing="ij")
        beam = np.exp(-(l_grid**2 + m_grid**2) / (2 * (3 * cell_size[1]) ** 2))
        aperture = _compute_aperture_fft(beam)
        support = np.abs(aperture) > 1e-3 * np.abs(aperture).max()
        assert np.allclose(np.angle(aperture[support]), 0, atol=1e-6)
