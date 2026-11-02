import numpy as np
import xarray as xr

from astrohack.utils.gridding import grid_beam
from astrohack.utils.imaging import (
    calculate_far_field_aperture,
    calculate_far_field_aperture_direct,
    parallactic_derotation,
)


def _cell_centred_map(n_side, cell, freq):
    """One map with a sample on every beam grid cell centre: a Gaussian beam plus random complex structure."""
    rng = np.random.default_rng(3)
    offsets = np.arange(n_side) - n_side // 2
    l_index, m_index = np.meshgrid(offsets, offsets, indexing="ij")
    lm = np.stack([-cell * l_index.ravel(), cell * m_index.ravel()], axis=1)
    beam = np.exp(-(l_index.ravel() ** 2 + m_index.ravel() ** 2) / 8.0)
    vis = beam[:, None, None] + 0.05 * (
        rng.standard_normal((lm.shape[0], freq.size, 4))
        + 1j * rng.standard_normal((lm.shape[0], freq.size, 4))
    )
    dataset = xr.Dataset(
        {
            "VIS": (("time", "chan", "pol"), vis),
            "WEIGHT": (("time", "chan", "pol"), np.ones(vis.shape)),
            "DIRECTIONAL_COSINES": (("time", "lm"), lm),
        },
        coords={
            "time": np.arange(lm.shape[0], dtype=float),
            "chan": freq,
            "pol": ["XX", "XY", "YX", "YY"],
        },
    )
    return xr.DataTree.from_dict({"map_0": dataset})


def test_direct_transform_equals_fft_of_cell_centred_samples():
    """With one sample on each cell centre and nearest-neighbour gridding, the FFT path is the discrete Fourier
    transform of the samples, so the direct transform must give the same aperture, rotation and scale included."""
    n_side, cell = 13, 1.0e-4
    freq = np.array([100.0e9, 100.05e9])
    ant_ddi_xdt = _cell_centred_map(n_side, cell, freq)
    sky_cell_size = np.array([-cell, cell])
    summary = {
        "beam": {},
        "spectral": {"channel width": 5.0e7, "number of channels": 2},
    }
    beam_grid, _, freq_axis, _, _, _, grid_corr, _, beam_normalization = grid_beam(
        ant_ddi_xdt=ant_ddi_xdt,
        grid_size=np.array([n_side, n_side]),
        sky_cell_size=sky_cell_size,
        avg_chan=True,
        chan_tol_fac=0.005,
        telescope=None,
        grid_interpolation_mode="nearest",
        observation_summary=summary,
        label="test",
    )
    assert not grid_corr
    beam_grid = parallactic_derotation(
        data=beam_grid, parallactic_angle_dict=ant_ddi_xdt
    )
    fft_aperture, u_axis, v_axis, _, wavelength = calculate_far_field_aperture(
        grid=beam_grid,
        padding_factor=4,
        freq=freq_axis,
        telescope=None,
        sky_cell_size=sky_cell_size,
        apply_grid_correction=False,
        label="test",
    )
    direct_aperture = calculate_far_field_aperture_direct(
        ant_ddi_xdt=ant_ddi_xdt,
        beam_normalization=beam_normalization,
        u_axis=u_axis,
        v_axis=v_axis,
        wavelength=wavelength,
        avg_chan=True,
        chan_tol_fac=0.005,
        n_beam_cells=n_side**2,
        label="test",
    )
    scale = np.max(np.abs(fft_aperture))
    np.testing.assert_allclose(direct_aperture, fft_aperture, rtol=0, atol=1e-9 * scale)
