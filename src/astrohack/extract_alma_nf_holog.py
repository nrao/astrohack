import pathlib

import toolviper.utils.parameter

from astrohack.core.extract_alma_nf_holog import process_alma_nf_asdm
from astrohack.io.holog_mds import AstrohackHologFile
from astrohack.utils.file import overwrite_file
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate()
def extract_alma_nf_holog(
    asdm_name: str,
    holog_name: str | None = None,
    time_smoothing_interval: float | None = None,
    amplitude_calibration: str = "linterp",
    phase_calibration: str = "linterp",
    calibration_cycle: int = 3,
    plot_calibration: bool = False,
    save_calibration: bool = False,
    overwrite: bool = False,
    zarr_format: int = 3,
) -> AstrohackHologFile:
    """ Extract ALMA near field holography data from an ASDM. Creates holography output file.

    :param asdm_name: Name of the ALMA near field holography ASDM.
    :type asdm_name: str

    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to the ASDM name with \
    *holog.zarr* extension.
    :type holog_name: str, optional

    :param time_smoothing_interval: Time bin, in seconds, onto which the pointing and total power data are averaged, \
    defaults to the largest of the pointing and total power sampling intervals.
    :type time_smoothing_interval: float, optional

    :param amplitude_calibration: Fit to the amplitude calibration subscans that is divided out of the data, "none" \
    means no amplitude calibration, defaults to "linterp".
    :type amplitude_calibration: str, optional

    :param phase_calibration: Fit to the phase calibration subscans that is subtracted from the data, "none" means \
    no phase calibration, defaults to "linterp".
    :type phase_calibration: str, optional

    :param calibration_cycle: Number of subscans from one calibration subscan to the next, defaults to 3.
    :type calibration_cycle: int, optional

    :param plot_calibration: Plot the calibrations to *<holog_name>_amplitude_cal.png* and \
    *<holog_name>_phase_cal.png*, defaults to False.
    :type plot_calibration: bool, optional

    :param save_calibration: Save the calibrated total power data to *<holog_name>.cal.zarr*, defaults to False.
    :type save_calibration: bool, optional

    :param overwrite: Overwrite holog file on disk, defaults to False.
    :type overwrite: bool, optional

    :param zarr_format: Zarr format of the output files, 2 or 3, defaults to 3
    :type zarr_format: int, optional

    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:

    The ASDM is read with `pyasdm <https://github.com/casangi/pyasdm>`_. The total power correlations are converted \
    to amplitude and phase, calibrated against the calibration subscans of each calibration cycle and averaged, \
    together with the pointing data, onto a common time axis.

    .. rubric:: Available calibration fits:
    - *linterp*:    Linear interpolation between calibration subscans
    - *sqinterp*:   Quadratic interpolation between calibration subscans
    - *spline*:     Spline fit to the calibration subscans
    - *regression*: Linear regression of the calibration subscans
    - *mean*:       Mean of the calibration subscans
    - *none*:       No calibration

    **Example Usage**

    .. parsed-literal::
        from astrohack import extract_alma_nf_holog, holog

        extract_alma_nf_holog(
            asdm_name="uid___A002_Xad52b4_X3",
            holog_name="nf_X3.holog.zarr",
        )

        holog(holog_name="nf_X3.holog.zarr", to_stokes=False)

    **AstrohackHologFile**

    The holog object contains a single antenna, DDI and map, `ant_<antenna name>/ddi_0/map_0`, with the attribute \
    `near_field` set, which `holog` uses to apply the near field corrections. Its polarization axis is \
    `["I", "R2"]` (R2 being the reference power), so `holog` must be called with `to_stokes=False`.
    """
    # Doing this here allows it to get captured by locals()
    holog_name = get_default_file_name(asdm_name, ".holog.zarr", holog_name)

    extract_params = locals()

    if not pathlib.Path(asdm_name, "ASDM.xml").is_file():
        raise FileNotFoundError(f"{asdm_name} is not an ASDM (no ASDM.xml found).")

    overwrite_file(holog_name, overwrite)
    holog_mds = AstrohackHologFile.create_from_input_parameters(
        holog_name, extract_params.copy(), zarr_format
    )

    process_alma_nf_asdm(extract_params, holog_mds)

    holog_mds.write(mode="w")
    holog_mds.open()
    return holog_mds
