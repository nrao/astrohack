import numpy as np
import xarray as xr

from astrohack.core.extract_holog import _drop_moving_time_samples


def test_drop_moving_time_samples():
    """Time samples whose pointing moves by more than lambda/D are dropped: a jump in the pointing and a slew."""
    pnt_time = np.round(np.arange(0.0, 10.0, 0.05), 2)
    l_axis = (
        1.0e-4 * pnt_time
    )  # a slow scan, 1e-5 rad per time sample against lambda/D = 2.5e-4 rad
    l_axis[pnt_time >= 6.0] = 0.0  # the slew back to the source
    m_axis = np.zeros_like(l_axis)
    m_axis[pnt_time == 3.0] = 1.0e-3  # one pointing sample off the scan
    pnt_mds = {
        "ant_DA41": xr.Dataset(
            {
                "DIRECTIONAL_COSINES": (
                    ("time", "lm"),
                    np.stack([l_axis, m_axis], axis=1),
                )
            },
            coords={"time": pnt_time},
        )
    }
    time_vis = np.round(np.arange(0.05, 9.0, 0.1), 2)
    used_samples_dict = {0: np.full(time_vis.size, True)}
    _drop_moving_time_samples(
        used_samples_dict,
        (0,),
        ["ant_DA41"],
        time_vis,
        0.1,
        pnt_mds,
        "ALMA",
        np.array([100.0e9]),
    )
    np.testing.assert_allclose(time_vis[~used_samples_dict[0]], [2.95, 3.05, 5.95])
