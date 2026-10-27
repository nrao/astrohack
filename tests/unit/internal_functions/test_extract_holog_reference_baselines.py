import numpy as np

from astrohack.core.extract_holog import _match_visibilities_to_time_bins


def test_all_reference_baselines_are_averaged():
    """A mapping antenna with two reference antennas gets the weighted mean of both baselines, the second one
    conjugated since the mapping antenna is its ANTENNA2; the baseline between the reference antennas is left out."""
    n_time = 10
    times = np.repeat(np.arange(n_time, dtype=float), 3)
    ant1 = np.tile([0, 2, 1], n_time)
    ant2 = np.tile([1, 0, 2], n_time)
    vis_data = np.tile(np.array([1 + 1j, 3 + 3j, 100 + 0j]), n_time)[:, None, None]
    weight = np.ones((times.size, 1))
    flag = np.zeros(vis_data.shape, dtype=bool)
    flag_row = np.zeros(times.size, dtype=bool)
    scan_list = np.ones(times.size, dtype=int)
    _, vis_map_dict, weight_map_dict, _, used_samples_dict, _, _ = (
        _match_visibilities_to_time_bins(
            vis_data,
            weight,
            ant1,
            ant2,
            times,
            flag,
            flag_row,
            ((1, 2),),
            (0,),
            1.0,
            scan_list,
        )
    )
    used = used_samples_dict[0]
    assert np.any(used)
    np.testing.assert_allclose(vis_map_dict[0][used, 0, 0], 2 - 1j)
    assert np.all(weight_map_dict[0][used, 0, 0] % 2 == 0)
