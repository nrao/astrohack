from math import factorial

import numpy as np
import pytest

from astrohack.utils.zernike_aperture_fitting import (
    _fit_an_aperture_plane_component_np_least_squares,
    create_osa_coordinates,
    zernike_matrix_functions,
)


def _textbook_zernike(n, m, x, y):
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    radial = sum(
        (-1) ** s
        * factorial(n - s)
        / (
            factorial(s)
            * factorial((n + abs(m)) // 2 - s)
            * factorial((n - abs(m)) // 2 - s)
        )
        * rho ** (n - 2 * s)
        for s in range((n - abs(m)) // 2 + 1)
    )
    return radial * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))


class TestZernikeMatrices:
    @pytest.mark.parametrize("order", range(11))
    def test_columns_are_the_labelled_polynomials(self, order):
        """Test that every column is the OSA labelled Zernike polynomial, with X = v and Y = u"""
        rng = np.random.default_rng(0)
        rho = np.sqrt(rng.uniform(0, 1, 500))
        theta = rng.uniform(0, 2 * np.pi, 500)
        u_ax, v_ax = rho * np.sin(theta), rho * np.cos(theta)
        matrix = zernike_matrix_functions[order](u_ax, v_ax)
        labels = create_osa_coordinates(order, split_nm=True)
        assert matrix.shape == (u_ax.size, len(labels))
        for column, (n, m) in enumerate(labels):
            assert np.allclose(
                matrix[:, column], _textbook_zernike(n, m, v_ax, u_ax), atol=1e-9
            ), f"N={n}, M={m}"

    def test_complex_coefficients_share_one_scale(self):
        """Test that the real and imaginary coefficients are both in units of the peak amplitude"""
        grid = np.linspace(-1, 1, 41)
        u_grid, v_grid = np.meshgrid(grid, grid, indexing="ij")
        inside = np.hypot(u_grid, v_grid) <= 1
        matrix = zernike_matrix_functions[2](u_grid[inside], v_grid[inside])
        aperture = (1 + 0.1j) * matrix[:, 4]
        max_ap = np.max(np.abs(aperture))
        real_coeffs, _, _ = _fit_an_aperture_plane_component_np_least_squares(
            matrix, aperture.real, max_ap
        )
        imag_coeffs, _, _ = _fit_an_aperture_plane_component_np_least_squares(
            matrix, aperture.imag, max_ap
        )
        assert np.isclose(real_coeffs[4] + 1j * imag_coeffs[4], (1 + 0.1j) / max_ap)
