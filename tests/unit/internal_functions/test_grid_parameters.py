import numpy as np
import pytest
import scipy.constants
import xarray as xr

from astrohack.utils.algorithms import calculate_optimal_grid_parameters


@pytest.mark.parametrize(
    "range_in_cells, n_pix", [(27.02, 27), (26.98, 27), (13.62, 13)]
)
def test_grid_cells_lie_inside_the_kept_samples(range_in_cells, n_pix):
    """The grid has an odd number of cells, centred on the raster, with every cell centre inside the kept samples;
    time bins that are not kept, such as a turnaround far off the raster, do not count."""
    diameter, freq = 12.0, np.array([100.0e9])
    cell = 0.85 * scipy.constants.speed_of_light / freq[0] / diameter
    half_range = range_in_cells * cell / 2
    l_axis = np.r_[np.linspace(-half_range, half_range, 101), -4 * half_range]
    m_axis = np.r_[np.linspace(half_range, -half_range, 101), 0.0]
    pnt_map_dict = {
        "ant_DA41": xr.Dataset(
            {"POINTING_OFFSET": (("time", "az_el"), np.stack([l_axis, m_axis], axis=1))}
        )
    }
    used_samples = np.r_[np.full(101, True), False]
    grid_size, cell_size = calculate_optimal_grid_parameters(
        pnt_map_dict, "ant_DA41", diameter, freq, 0, used_samples
    )
    assert grid_size == [n_pix, n_pix]
    assert (n_pix // 2) * cell_size <= half_range
