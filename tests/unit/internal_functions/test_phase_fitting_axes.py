import numpy as np

from astrohack.utils.constants import clight
from astrohack.utils.phase_fitting import _clic_full_phase_fitting


class TestClicPhaseFittingAxes:
    frequency = 104.02e9
    focus = 4.8
    defocus = 0.0988
    # Ordered as holog's u_prime and v_prime
    u_axis = np.linspace(6.5, -6.5, 131)
    v_axis = np.linspace(-6.5, 6.5, 131)

    def test_axis_order(self):
        """Test that the fitted tilts, lateral feed offsets and astigmatism refer to the axes of the (u, v) ordered
        phase"""
        wave_number = 2 * np.pi * self.frequency / clight
        u_grid, v_grid = np.meshgrid(self.u_axis, self.v_axis, indexing="ij")
        radius2 = u_grid**2 + v_grid**2
        focus_term = 1 - radius2 / (4 * self.focus**2) + self.defocus / self.focus
        lateral = 1 / (1 + self.defocus / self.focus) - 1 / np.sqrt(
            radius2 / self.focus**2 + focus_term**2
        )
        # Constant, u and v tilts, z, x and y feed offsets and astigmatism, small enough for the phase not to wrap
        parameters = np.array(
            [
                0.1,
                0.04,
                -0.02,
                0.0,
                0.3e-3 * wave_number,
                -0.2e-3 * wave_number,
                0.2,
                0.0,
            ]
        )
        phase = (
            parameters[0]
            + parameters[1] * u_grid
            + parameters[2] * v_grid
            + parameters[4] * u_grid / self.focus * lateral
            + parameters[5] * v_grid / self.focus * lateral
            + parameters[6] * (u_grid**2 - v_grid**2) / (2 * self.focus**2)
        )
        assert np.abs(phase).max() < np.pi

        best_fit, _ = _clic_full_phase_fitting(
            8,
            self.frequency,
            12.0,
            0.375,
            self.focus,
            self.defocus,
            phase,
            self.u_axis,
            self.v_axis,
        )
        assert np.allclose(best_fit, parameters, rtol=0, atol=1e-6)
