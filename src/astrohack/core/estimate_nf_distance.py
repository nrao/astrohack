import os
import tempfile

import numpy as np
import toolviper.utils.logger as logger

from astrohack.holog import holog
from astrohack.io.dio import open_image
from astrohack.utils.constants import clight


def scan_nf_distance(estimate_params: dict) -> dict:
    """
    Image a near field holog file at each trial distance and measure the surface rms that holog's phase fit leaves
    in an annulus of the aperture.

    Args:
        estimate_params: input parameters of estimate_nf_distance

    Returns:
        trial distances and residual surface rms in micrometers, and the distance of the smallest residual
    """
    distances = np.sort(np.asarray(estimate_params["distances"], dtype=float))
    residuals = np.empty(distances.size)
    with tempfile.TemporaryDirectory() as work_dir:
        for i_dist, distance in enumerate(distances):
            image_name = os.path.join(work_dir, f"trial_{i_dist}.image.zarr")
            holog(
                holog_name=estimate_params["holog_name"],
                image_name=image_name,
                nf_distance=float(distance),
                nf_focus_offset=estimate_params["focus_offset"],
                to_stokes=False,
                parallel=False,
                overwrite=True,
            )
            residuals[i_dist] = _residual_surface_rms(
                image_name,
                estimate_params["inner_radius"],
                estimate_params["outer_radius"],
            )
            logger.info(
                f"Trial distance {distance:.2f} m: residual surface rms {residuals[i_dist]:.2f} um"
            )

    best, best_residual, converged = _fit_minimum(
        distances, residuals, estimate_params["converged_factor"]
    )
    return {
        "distance": best,
        "residual": best_residual,
        "distances": distances.tolist(),
        "residuals": residuals.tolist(),
        "converged": converged.tolist(),
    }


def _residual_surface_rms(image_name, inner_radius, outer_radius):
    """Rms in micrometers of the surface left by the phase fit, between the inner and outer radius."""
    image_mds = open_image(image_name)
    ant_key = [key for key in image_mds.keys() if key.startswith("ant_")][0]
    xds = image_mds[ant_key]["ddi_0"]
    phase = xds.CORRECTED_PHASE.values[0, 0, 0]
    radius = np.hypot(xds.u_prime.values[:, None], xds.v_prime.values[None, :])
    ring = (radius > inner_radius) & (radius < outer_radius)
    wavelength = clight / float(np.mean(xds.chan.values))
    return float(np.sqrt(np.nanmean(phase[ring] ** 2)) * wavelength / (4 * np.pi) * 1e6)


def _fit_minimum(distances, residuals, converged_factor):
    """
    Distance of the smallest residual from a parabola in the squared residual through the converged trials, those
    within converged_factor of the smallest residual. The squared residual is a parabola in the distance when the
    residual phase grows linearly with the distance error. Falls back to the trial of the smallest residual when
    fewer than three trials converged or the parabola has no minimum inside them.
    """
    converged = residuals < converged_factor * np.nanmin(residuals)
    i_min = int(np.nanargmin(residuals))
    if converged.sum() >= 3:
        x_conv = distances[converged]
        a_coeff, b_coeff, c_coeff = np.polyfit(x_conv, residuals[converged] ** 2, 2)
        if a_coeff > 0 and x_conv.min() <= -b_coeff / (2 * a_coeff) <= x_conv.max():
            best = -b_coeff / (2 * a_coeff)
            return (
                float(best),
                float(np.sqrt(np.polyval([a_coeff, b_coeff, c_coeff], best))),
                converged,
            )
    logger.warning(
        "The residuals give no parabolic minimum; using the trial distance of the smallest residual"
    )
    return float(distances[i_min]), float(residuals[i_min]), converged
