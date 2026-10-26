import numpy as np

from astrohack.utils.algorithms import phase_wrapping
from astrohack.utils.constants import clight
from astrohack.utils.phase_fitting import (
    _build_astigmatism_matrix,
    _clic_full_phase_fitting,
)


class TestClicPhaseFitting:
    frequency = 104.02e9
    diameter = 12.0
    blockage = 0.75
    focus = 4.8
    defocus = 0.091
    axis = np.linspace(-6.5, 6.5, 131)

    def test_wrapped_phase(self):
        """Test that the CLIC-like fit recovers a phase model that wraps inside the aperture"""
        wave_number = 2 * np.pi * self.frequency / clight
        tilt_step = 1e-3 / (self.diameter / 2) * wave_number
        # Constant, u and v tilts, z, x and y feed offsets and astigmatism, with the tilts and feed offsets close to
        # points of the fit's 1 mm trial grid
        parameters = np.array(
            [
                0.4,
                1.1 * tilt_step,
                -0.9 * tilt_step,
                1.2e-3 * wave_number,
                0.9e-3 * wave_number,
                -0.3e-3 * wave_number,
                0.5,
                -0.3,
            ]
        )
        matrix, _, sel = _build_astigmatism_matrix(
            np.zeros((self.axis.size, self.axis.size)),
            self.axis,
            self.axis,
            self.focus,
            self.defocus,
            self.diameter,
            self.blockage,
            8,
            np.pi,
        )
        model = matrix @ parameters
        assert np.ptp(model[sel]) > 2 * np.pi

        best_fit, phase_model = _clic_full_phase_fitting(
            8,
            self.frequency,
            self.diameter,
            self.blockage,
            self.focus,
            self.defocus,
            phase_wrapping(model),
            self.axis,
            self.axis,
        )
        assert np.allclose(best_fit, parameters, rtol=0, atol=1e-6)
        assert np.allclose(phase_wrapping(model - phase_model)[sel], 0, atol=1e-6)
