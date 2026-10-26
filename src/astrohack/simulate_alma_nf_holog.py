import toolviper.utils.parameter
import xarray as xr

from astrohack.core.simulate_alma_nf_holog import simulate_alma_nf_map
from astrohack.io.holog_mds import AstrohackHologFile
from astrohack.utils.file import overwrite_file


@toolviper.utils.parameter.validate()
def simulate_alma_nf_holog(
    holog_name: str,
    distance: float | None = None,
    focus_offset: float | None = None,
    geometry: str = "rotating",
    antenna_name: str = "DV08",
    alma_osf_pad: str = "TF04",
    frequency: float = 104.02e9,
    map_size: float = 0.0215,
    map_cell: float = 1.6e-4,
    template_holog_name: str | None = None,
    aperture_cell: float = 0.06,
    feed_legs: bool = True,
    surface_bumps: list[list[float]] | None = None,
    overwrite: bool = False,
    zarr_format: int = 3,
) -> AstrohackHologFile:
    """ Simulate an ALMA near field holography map and write it as a holog file.

    :param holog_name: Name of the *<holog_name>.holog.zarr* file to create.
    :type holog_name: str

    :param distance: Distance in meters from the aperture centre to the transmitter, defaults to the distance \
    tabulated for the pad.
    :type distance: float, optional

    :param focus_offset: Axial offset in meters of the holography receiver from the focus, defaults to the offset \
    tabulated for the pad.
    :type focus_offset: float, optional

    :param geometry: "rotating": the antenna turns about its elevation axis while it maps the transmitter, as \
    ALMA antennas do; "fixed": the transmitter stays at the same distance from the aperture centre, as in the \
    model of Baars et al. (2007), defaults to "rotating".
    :type geometry: str, optional

    :param antenna_name: Name of the simulated antenna, which sets its optics, defaults to "DV08".
    :type antenna_name: str, optional

    :param alma_osf_pad: OSF pad recorded as the antenna's station, defaults to "TF04".
    :type alma_osf_pad: str, optional

    :param frequency: Frequency of the transmitter in Hz, defaults to 104.02e9.
    :type frequency: float, optional

    :param map_size: Width in radians of the square raster of direction cosines, defaults to 0.0215.
    :type map_size: float, optional

    :param map_cell: Spacing in radians of the raster, defaults to 1.6e-4.
    :type map_cell: float, optional

    :param template_holog_name: Near field holog file whose antenna, pad, frequency, direction cosines and times are \
    used instead of the raster, defaults to None.
    :type template_holog_name: str, optional

    :param aperture_cell: Spacing in meters of the aperture points summed over, defaults to 0.06.
    :type aperture_cell: float, optional

    :param feed_legs: Block the aperture along the shadows of the feed legs, defaults to True.
    :type feed_legs: bool, optional

    :param surface_bumps: Gaussian surface deformations, each [x, y, sigma, height] in meters along the surface \
    normal, defaults to None.
    :type surface_bumps: list, optional

    :param overwrite: Overwrite holog file on disk, defaults to False.
    :type overwrite: bool, optional

    :param zarr_format: Zarr format of the output file, 2 or 3, defaults to 3
    :type zarr_format: int, optional

    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:

    The aperture field is that of the near field holography feed, with the path change from the receiver's \
    axial offset, on the antenna's dish minus its central blockage and, optionally, the feed leg shadows. The \
    response to the transmitter is summed over the aperture along the exact path from the transmitter to each \
    aperture point, and normalised to a peak of 1. The file has the layout of `extract_alma_nf_holog`'s, \
    `ant_<antenna name>/ddi_0/map_0` with the `near_field` attribute, and the simulation parameters in the \
    map's `simulation` attribute, so `holog` processes it as measured data.

    **Example Usage**

    .. parsed-literal::
        from astrohack import holog, simulate_alma_nf_holog

        simulate_alma_nf_holog(holog_name="sim.holog.zarr", distance=280.0)

        holog(holog_name="sim.holog.zarr", to_stokes=False)
    """
    sim_params = locals()

    overwrite_file(holog_name, overwrite)
    holog_mds = AstrohackHologFile.create_from_input_parameters(
        holog_name, sim_params.copy(), zarr_format
    )

    xds, antenna, _, _ = simulate_alma_nf_map(sim_params)
    holog_mds.root[f"ant_{antenna}/ddi_0/map_0"] = xr.DataTree(dataset=xds)

    holog_mds.write(mode="w")
    holog_mds.open()
    return holog_mds
