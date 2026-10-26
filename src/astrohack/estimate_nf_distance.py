import numpy as np
import toolviper.utils.parameter

from astrohack.core.estimate_nf_distance import scan_nf_distance


@toolviper.utils.parameter.validate()
def estimate_nf_distance(
    holog_name: str,
    distances: list[float] | np.ndarray,
    focus_offset: float | None = None,
    inner_radius: float = 1.0,
    outer_radius: float = 5.5,
    converged_factor: float = 3.0,
) -> dict:
    """ Estimate the distance to the transmitter of a near field holography from the data.

    :param holog_name: Name of the near field *<holog_name>.holog.zarr* file.
    :type holog_name: str

    :param distances: Trial distances in meters from the aperture centre to the transmitter.
    :type distances: list, numpy.ndarray

    :param focus_offset: Axial offset in meters of the holography receiver from the focus, defaults to the offset \
    tabulated for the pad.
    :type focus_offset: float, optional

    :param inner_radius: Inner radius in meters of the aperture annulus where the residual is measured, defaults to \
    1.0.
    :type inner_radius: float, optional

    :param outer_radius: Outer radius in meters of the aperture annulus, defaults to 5.5.
    :type outer_radius: float, optional

    :param converged_factor: Trials whose residual is within this factor of the smallest one enter the fit, \
    defaults to 3.0.
    :type converged_factor: float, optional

    :return: The distance of the smallest residual ("distance"), the residual surface rms there in micrometers \
    ("residual"), and the trial distances, their residuals and which of them entered the fit ("distances", \
    "residuals", "converged").
    :rtype: dict

    .. _Description:

    The near field corrections depend on the distance, so an aperture imaged at the wrong distance keeps a residual \
    phase that holog's phase fit cannot remove. Each trial distance is imaged \
    with `holog` (`nf_distance`, `to_stokes=False`), and the surface rms left by the phase fit is measured in the \
    annulus. The distance of the smallest residual comes from a parabola in the squared residual through the trials \
    within `converged_factor` of the smallest residual. Trials far from the true distance, where the phase fit does \
    not converge, leave residuals of hundreds of micrometers and are left out.

    The trial images are written to a temporary directory and removed. The estimate inherits the geometry of \
    holog's near field model: on data simulated with `simulate_alma_nf_holog`, it is within a few tenths of a meter \
    of the true distance when the antenna rotates about its elevation axis, and a few meters too far with the \
    fixed-distance geometry of Baars et al. (2007).

    **Example Usage**

    .. parsed-literal::
        import numpy as np
        from astrohack import estimate_nf_distance

        result = estimate_nf_distance(holog_name="nf_X3.holog.zarr", distances=np.arange(266.0, 291.0, 2.0))
        print(result["distance"], result["residual"])
    """
    estimate_params = locals()
    return scan_nf_distance(estimate_params)
