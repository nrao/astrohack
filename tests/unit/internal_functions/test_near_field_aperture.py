import numpy as np

from astrohack import get_proper_telescope
from astrohack.utils.imaging import calculate_near_field_aperture


def test_near_field_amplitude_without_apodization():
    """The apodization of the beam changes the near field aperture phase, not its amplitude."""
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(1, 1, 1, 32, 32)) + 1j * rng.normal(size=(1, 1, 1, 32, 32))
    params = {
        "sky_cell_size": np.array([1.6e-4, 1.6e-4]),
        "distance": 278.6315,
        "freq": np.array([104.02e9]),
        "padding_factor": 4,
        "focus_offset": 0.0988,
        "telescope": get_proper_telescope("alma", "DV08"),
        "apply_grid_correction": False,
        "label": "test",
    }
    apodized = calculate_near_field_aperture(grid, apodize=True, **params)[0]
    plain = calculate_near_field_aperture(grid, apodize=False, **params)[0]
    np.testing.assert_allclose(np.abs(apodized[0, 0, 0]), np.abs(plain[0, 0, 0]))
    assert not np.allclose(np.angle(apodized[0, 0, 0]), np.angle(plain[0, 0, 0]))
