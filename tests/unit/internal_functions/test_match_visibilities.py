import numpy as np

from astrohack.core.extract_holog import _match_visibilities_to_time_bins

# XX, XY, YX, YY of one baseline
VIS = np.array([1.0 + 0.5j, 0.2 + 0.3j, -0.4 + 0.1j, 0.9 - 0.2j])
WEIGHT = np.array([1.0, 2.0, 3.0, 4.0])


def _match(vis, weight, flag, pol_transpose):
    # mapping antenna 1 is the second antenna of its baseline with reference 0, mapping antenna 2 the first of its
    # baseline with reference 3; two integrations each, in one time bin
    return _match_visibilities_to_time_bins(
        vis,
        weight,
        np.array([0, 0, 2, 2]),
        np.array([1, 1, 3, 3]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        flag,
        np.zeros(4, dtype=bool),
        ((0,), (3,)),
        (1, 2),
        1.0,
        np.ones(4, dtype=int),
        pol_transpose,
    )


def test_conjugated_rows_are_transposed():
    flag = np.zeros((4, 1, 4), dtype=bool)
    flag[0, 0, 2] = True
    _, vis_map, weight_map, *_ = _match(
        np.tile(VIS, (4, 1, 1)), np.tile(WEIGHT, (4, 1)), flag, np.array([0, 2, 1, 3])
    )
    np.testing.assert_allclose(vis_map[2][0, 0], VIS)
    np.testing.assert_allclose(weight_map[2][0, 0], 2 * WEIGHT)
    # conjugating a baseline transposes its correlation matrix: XY of (1, 0) is the conjugate of YX of (0, 1),
    # with YX's weight and flag
    np.testing.assert_allclose(vis_map[1][0, 0], np.conjugate(VIS[[0, 2, 1, 3]]))
    np.testing.assert_allclose(weight_map[1][0, 0], [2.0, 3.0, 4.0, 8.0])


def test_unpaired_cross_hand_of_conjugated_rows_is_flagged():
    # XX, XY, YY: a conjugated row has no YX to give XY
    vis = np.tile(VIS[[0, 1, 3]], (4, 1, 1))
    _, vis_map, weight_map, *_ = _match(
        vis, np.ones((4, 3)), np.zeros((4, 1, 3), dtype=bool), np.array([0, -1, 2])
    )
    np.testing.assert_allclose(vis_map[2][0, 0], VIS[[0, 1, 3]])
    np.testing.assert_allclose(
        vis_map[1][0, 0], [np.conjugate(VIS[0]), 0, np.conjugate(VIS[3])]
    )
    np.testing.assert_allclose(weight_map[1][0, 0], [2.0, 0.0, 2.0])
