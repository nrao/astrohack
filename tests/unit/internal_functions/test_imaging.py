import numpy as np
import pytest

from astrohack.utils.imaging import parallactic_derotation


class TestParallacticDerotation:
    @pytest.mark.parametrize("size", [28, 31])
    def test_rotation_centre(self, size):
        """Test that the beam grid is rotated by 90 degrees about index N // 2, the FFT origin"""
        centre = size // 2
        beam = np.zeros([1, 1, 1, size, size], dtype=complex)
        beam[0, 0, 0, centre + 2, centre + 1] = 1.0
        rotated = parallactic_derotation(beam, {"map_0": None})
        assert np.isclose(rotated[0, 0, 0, centre - 1, centre + 2], 1.0)
