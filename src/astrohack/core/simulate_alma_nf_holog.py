import numpy as np
import xarray as xr

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.core.extract_alma_nf_holog import _data_to_xds
from astrohack.io.dio import open_holog
from astrohack.utils.constants import clight
from astrohack.utils.imaging import FEED_GE_TAB, FEED_GH_TAB

DAY2SEC = 86400
LEG_HALF_WIDTH = 0.075
SAMPLE_INTERVAL = 0.048
START_MJD = 59000.0


def simulate_alma_nf_map(sim_params: dict) -> tuple[xr.Dataset, str, float, float]:
    """
    Simulate an ALMA near field holography map: the response of a model aperture to a transmitter at a finite
    distance, sampled on a raster of direction cosines.

    Args:
        sim_params: input parameters of simulate_alma_nf_holog

    Returns:
        map dataset in the layout of extract_alma_nf_holog, antenna name, distance and focus offset used
    """
    template = sim_params["template_holog_name"]
    if template is None:
        antenna, station = sim_params["antenna_name"], sim_params["alma_osf_pad"]
        frequency = sim_params["frequency"]
        lm = _regular_raster(sim_params["map_size"], sim_params["map_cell"])
        time = START_MJD + np.arange(lm.shape[0]) * SAMPLE_INTERVAL / DAY2SEC
        integ_time = SAMPLE_INTERVAL / DAY2SEC
    else:
        antenna, station, frequency, lm, time, integ_time = _template_raster(template)

    telescope = get_proper_telescope("ALMA", antenna)
    distance, focus_offset = sim_params["distance"], sim_params["focus_offset"]
    if distance is None or focus_offset is None:
        pad_distance, pad_focus_offset = telescope.station_distance_dict[station]
        distance = pad_distance if distance is None else distance
        focus_offset = pad_focus_offset if focus_offset is None else focus_offset

    wavenumber = 2 * np.pi * frequency / clight
    x_ap, y_ap, field = _aperture_field(
        telescope,
        focus_offset,
        wavenumber,
        sim_params["aperture_cell"],
        sim_params["feed_legs"],
        sim_params["surface_bumps"],
    )
    response = _transmitter_response(
        lm, x_ap, y_ap, field, distance, telescope, wavenumber, sim_params["geometry"]
    )
    response /= np.abs(response).max()

    n_samp = lm.shape[0]
    combined_data = {
        "time": time,
        "amp": np.abs(response),
        "pha": np.angle(response),
        "ref": np.ones(n_samp),
        "weight": np.ones(n_samp),
        "lm": lm,
        "offset": lm,
        "integ_time": integ_time,
        "az_el_info": {"center": [0.0, 0.0], "mean": [0.0, 0.0], "median": [0.0, 0.0]},
    }
    meta_dict = {
        "spw": {"frequency": np.array([frequency]), "bandwidth": 0.0, "nchan": 1},
        "field": {"name": "Simulated transmitter", "phase_dir": [0.0, 0.0]},
        "ant": {"telescope": "ALMA", "antenna": antenna, "station": station},
        "holo": {"focus": focus_offset},
    }
    xds = _data_to_xds(combined_data, meta_dict)
    xds.attrs["simulation"] = {
        "distance": distance,
        "focus_offset": focus_offset,
        "geometry": sim_params["geometry"],
        "feed_legs": sim_params["feed_legs"],
        "surface_bumps": sim_params["surface_bumps"],
    }
    return xds, antenna, distance, focus_offset


def _regular_raster(map_size, map_cell):
    """Square raster of direction cosines, map_size wide, with map_cell spacing."""
    axis = np.arange(-map_size / 2, map_size / 2 + map_cell / 2, map_cell)
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([l_grid.ravel(), m_grid.ravel()])


def _template_raster(template):
    """Antenna, pad, frequency, direction cosines and time of the map of a near field holog file."""
    holog_mds = open_holog(template)
    ant_key = [key for key in holog_mds.keys() if key.startswith("ant_")][0]
    xds = holog_mds[ant_key]["ddi_0"]["map_0"].dataset
    general = xds.attrs["summary"]["general"]
    return (
        general["antenna name"],
        general["station"],
        float(xds.chan.values[0]),
        xds.DIRECTIONAL_COSINES.values,
        xds.time.values / DAY2SEC,
        xds.attrs["time_smoothing_interval"] / DAY2SEC,
    )


def _aperture_field(
    telescope, focus_offset, wavenumber, aperture_cell, feed_legs, surface_bumps
):
    """
    Complex field on the aperture plane, halfway between vertex and rim: the feed illumination, the path change
    from the axial feed offset, and the surface deformations.
    """
    radius, focal = telescope.diameter / 2, telescope.focus
    axis = np.arange(-radius, radius + aperture_cell / 2, aperture_cell)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    r_grid = np.hypot(x_grid, y_grid)
    inside = (r_grid <= radius) & (r_grid >= telescope.inner_radial_limit)
    if feed_legs:
        inside &= (np.abs(x_grid) > LEG_HALF_WIDTH) & (np.abs(y_grid) > LEG_HALF_WIDTH)
    x_ap, y_ap, r_ap = x_grid[inside], y_grid[inside], r_grid[inside]

    # E plane along x: with this orientation holog's feed correction removes the feed phase exactly.
    psi = 2 * np.arctan(r_ap / (2 * focal))
    cos_k = np.cos(np.outer(psi, np.arange(1, len(FEED_GE_TAB) + 1)))
    phi = np.arctan2(y_ap, x_ap)
    gain = (cos_k @ np.array(FEED_GE_TAB)) * np.cos(phi) ** 2 + (
        cos_k @ np.array(FEED_GH_TAB)
    ) * np.sin(phi) ** 2
    amplitude = np.abs(gain) / (1 + r_ap**2 / (4 * focal**2))

    path_change = np.sqrt(
        r_ap**2 + (focal + focus_offset - r_ap**2 / (4 * focal)) ** 2
    ) - (focal + r_ap**2 / (4 * focal) + focus_offset)
    phase = -wavenumber * path_change - np.angle(gain)
    if surface_bumps:
        displacement = np.zeros_like(r_ap)
        for x_0, y_0, sigma, height in surface_bumps:
            displacement += height * np.exp(
                -((x_ap - x_0) ** 2 + (y_ap - y_0) ** 2) / (2 * sigma**2)
            )
        phase += 2 * wavenumber * displacement / np.sqrt(1 + r_ap**2 / (4 * focal**2))
    return x_ap, y_ap, amplitude * np.exp(1j * phase)


def _transmitter_response(
    lm, x_ap, y_ap, field, distance, telescope, wavenumber, geometry, chunk=256
):
    """
    Response to a transmitter at a finite distance, summed over the aperture along the exact paths. With the
    "rotating" geometry the antenna turns about its elevation axis, Delta behind the aperture centre; with the
    "fixed" geometry of Baars et al. (2007) the transmitter stays at the distance from the aperture centre.
    """
    z_max = (telescope.diameter / 2) ** 2 / 4 / telescope.focus
    delta = telescope.el_axis_off + z_max / 2
    l_p, m_p = lm[:, 0], lm[:, 1]
    if geometry == "rotating":
        n_p = np.sqrt(1 - l_p**2 - m_p**2)
        t_x, t_y = (distance + delta) * l_p, (distance + delta) * m_p
        t_z = (distance + delta) * n_p - delta
    else:
        l_u, m_v = l_p * (1 + delta / distance), m_p * (1 + delta / distance)
        t_x, t_y = distance * l_u, distance * m_v
        t_z = distance * np.sqrt(1 - l_u**2 - m_v**2)

    response = np.empty(lm.shape[0], dtype=complex)
    for i_0 in range(0, lm.shape[0], chunk):
        i_1 = min(i_0 + chunk, lm.shape[0])
        path = np.sqrt(
            (t_x[i_0:i_1, None] - x_ap) ** 2
            + (t_y[i_0:i_1, None] - y_ap) ** 2
            + t_z[i_0:i_1, None] ** 2
        )
        response[i_0:i_1] = np.exp(-1j * wavenumber * (path - distance)) @ field
    return response
