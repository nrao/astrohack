import numpy as np

from astrohack.core.image_comparison_tool import extract_rms_from_xds
from astrohack.utils import (
    dynamic_format,
    format_observation_summary,
    make_header,
    create_dataset_label,
)
from astrohack.antenna import AntennaSurface
from astrohack.utils import (
    convert_unit,
    clight,
    format_value_error,
    format_frequency,
    format_wavelength,
    format_value_unit,
    length_units,
    trigo_units,
    format_label,
    create_pretty_table,
    string_to_ascii_file,
)
import toolviper.utils.logger as logger

from astrohack.utils.phase_fitting import aips_par_names


def export_screws_chunk(parm_dict):
    """
    Chunk function for the user facing function export_screws
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    export_name = parm_dict["destination"] + f"/panel_screws_{antenna}_{ddi}."
    xds = parm_dict["xds_data"]
    surface = AntennaSurface(xds, reread=True)
    surface.export_screws(export_name + "txt", unit=parm_dict["unit"])
    surface.plot_screw_adjustments(export_name + "png", parm_dict)


def export_gains_table_chunk(parm_dict):
    in_waves = parm_dict["wavelengths"]
    in_freqs = parm_dict["frequencies"]
    ant = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    xds = parm_dict["xds_data"]
    antenna = AntennaSurface(xds, reread=True)
    frequency = clight / antenna.wavelength

    if in_waves is None and in_freqs is None:
        try:
            wavelengths = antenna.telescope.gain_wavelengths
        except AttributeError:
            msg = f"Telescope {antenna.telescope.name} has no predefined list of wavelengths to compute gains"
            logger.error(msg)
            logger.info("Please provide one in the arguments")
            raise Exception(msg)
    else:
        wave_fac = convert_unit(parm_dict["wavelength_unit"], "m", "length")
        freq_fac = convert_unit(parm_dict["frequency_unit"], "Hz", "frequency")
        wavelengths = []
        if in_waves is not None:
            if isinstance(in_waves, float) or isinstance(in_waves, int):
                in_waves = [in_waves]
            for in_wave in in_waves:
                wavelengths.append(wave_fac * in_wave)
        if in_freqs is not None:
            if isinstance(in_freqs, float) or isinstance(in_freqs, int):
                in_freqs = [in_freqs]
            for in_freq in in_freqs:
                wavelengths.append(clight / freq_fac / in_freq)

    db = "dB"
    rmsunit = parm_dict["rms_unit"]
    rmses = antenna.get_rms(rmsunit)

    field_names = [
        "Frequency",
        "Wavelength",
        "Before panel",
        "After panel",
        "Theoretical Max.",
    ]
    table = create_pretty_table(field_names)

    outstr = (
        f'# Gain estimates for {antenna.telescope.name} antenna {ant.split("_")[1]}\n'
    )
    outstr += f"# Based on a measurement at {format_frequency(frequency)}, {format_wavelength(antenna.wavelength)}\n"
    outstr += f"# Antenna surface RMS before adjustment: {format_value_unit(rmses[0], rmsunit)}\n"
    outstr += f"# Antenna surface RMS after adjustment: {format_value_unit(rmses[1], rmsunit)}\n"
    outstr += 1 * "\n"

    for wavelength in wavelengths:
        prior, theo = antenna.gain_at_wavelength(False, wavelength)
        after, _ = antenna.gain_at_wavelength(True, wavelength)
        row = [
            format_frequency(clight / wavelength),
            format_wavelength(wavelength),
            format_value_unit(prior, db),
            format_value_unit(after, db),
            format_value_unit(theo, db),
        ]
        table.add_row(row)

    outstr += table.get_string()
    string_to_ascii_file(
        outstr, parm_dict["destination"] + f"/panel_gains_{ant}_{ddi}.txt"
    )


def export_phase_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    phase_fit_results = parm_dict["xds_data"].attrs["phase_fitting"]
    if phase_fit_results is None:
        logger.warning(
            f"No phase fit results to export for {create_dataset_label(antenna, ddi)}"
        )
        return

    angle_unit = parm_dict["angle_unit"]
    length_unit = parm_dict["length_unit"]
    field_names = ["Parameter", "Value", "Unit"]
    alignment = ["l", "r", "c"]
    outstr = ""

    for mapkey, map_dict in phase_fit_results.items():
        for freq, freq_dict in map_dict.items():
            for pol, pol_dict in freq_dict.items():
                outstr += (
                    f'* {mapkey.replace("_", " ")}, Frequency {format_frequency(freq)}, '
                    f"polarization state {pol}:\n\n "
                )
                table = create_pretty_table(field_names, alignment)
                for par_name in aips_par_names:
                    item = pol_dict[par_name]
                    val = item["value"]
                    err = item["error"]
                    unit = item["unit"]
                    if unit in length_units:
                        fac = convert_unit(unit, length_unit, "length")
                    elif unit in trigo_units:
                        fac = convert_unit(unit, angle_unit, "trigonometric")
                    else:
                        msg = f"Unknown unit {unit}"
                        logger.error(msg)
                        raise Exception(msg)

                    row = [
                        format_label(par_name),
                        format_value_error(fac * val, fac * err, 1.0, 1e-4),
                        unit,
                    ]
                    table.add_row(row)

                outstr += table.get_string() + "\n\n"

    string_to_ascii_file(outstr, f"{destination}/image_phase_fit_{antenna}_{ddi}.txt")


def export_zernike_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    zernike_coeffs = parm_dict["xds_data"]["ZERNIKE_COEFFICIENTS"].values
    rms = parm_dict["xds_data"]["ZERNIKE_FIT_RMS"].values
    corr_axis = parm_dict["xds_data"].orig_pol.values
    freq_axis = parm_dict["xds_data"].chan.values
    ntime = zernike_coeffs.shape[0]
    osa_indices = parm_dict["xds_data"].osa.values
    destination = parm_dict["destination"]

    field_names = ["Indices", "Real", "Imaginary"]
    alignment = ["l", "c", "c"]
    outstr = ""

    for itime in range(ntime):
        for ichan, freq in enumerate(freq_axis):
            for icorr, corr in enumerate(corr_axis):
                outstr += f"* map {itime}, Frequency {format_frequency(freq)}, Correlation {corr}:\n"
                outstr += (
                    f"   Fit RMS = {rms[itime, ichan, icorr].real:.8f} + {rms[itime, ichan, icorr].imag:.8f}*i"
                    f"\n\n"
                )
                table = create_pretty_table(field_names, alignment)
                for icoeff, coeff in enumerate(zernike_coeffs[itime, ichan, icorr]):
                    row = [
                        osa_indices[icoeff],
                        f"{coeff.real:.8f}",
                        f"{coeff.imag:.8f}",
                    ]
                    table.add_row(row)

                outstr += table.get_string() + "\n\n"

    string_to_ascii_file(outstr, f"{destination}/image_zernike_fit_{antenna}_{ddi}.txt")


def create_fits_comparison_rms_table(parameters, xdt):
    image_list = xdt.children
    rms_unit = parameters["rms_unit"]

    fields = [
        "Image",
        "Reference",
        f"Original RMS [{rms_unit}]",
        f"Resampled RMS [{rms_unit}]",
        f"Reference RMS [{rms_unit}]",
        f"Residuals RMS [{rms_unit}]",
    ]

    factor = convert_unit("m", rms_unit, "length")

    table = create_pretty_table(fields)
    for image in image_list:

        image_xds = xdt[image]["Image"].to_dataset()
        reference_xds = xdt[image]["Reference"].to_dataset()

        img_rms_dict = extract_rms_from_xds(image_xds)
        ref_rms_dict = extract_rms_from_xds(reference_xds)
        values = np.array(
            [
                img_rms_dict["original"],
                img_rms_dict["resampled"],
                ref_rms_dict["original"],
                img_rms_dict["residuals"],
            ]
        )
        values *= factor

        row = [image_xds.attrs["filename"], reference_xds.attrs["filename"]]
        for val in values:
            row.append(f"{val:{dynamic_format(val)}}")

        table.add_row(row)

    outstr = f'RMS comparison table from {parameters["zarr_data_tree"]}:\n'
    outstr += table.get_string()
    string_to_ascii_file(outstr, parameters["table_file"])
    if parameters["print_table"]:
        print(table)
    return


def generate_observation_summary(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    try:
        map_id = parm_dict["this_map"]
        is_holog_zarr = True
    except KeyError:
        map_id = None
        is_holog_zarr = False

    xds = parm_dict["xdt_data"]
    obs_sum = xds.attrs["summary"]

    tab_size = parm_dict["tab_size"]
    tab_count = 1

    if is_holog_zarr:
        header = f"{antenna}, {ddi}, {map_id}"
    else:
        header = f"{antenna}, {ddi}"

    outstr = make_header(header, "#", 60, 3)

    outstr += (
        format_observation_summary(
            obs_sum,
            tab_size,
            tab_count,
            az_el_key=parm_dict["az_el_key"],
            phase_center_unit=parm_dict["phase_center_unit"],
            az_el_unit=parm_dict["az_el_unit"],
            time_format=parm_dict["time_format"],
        )
        + "\n"
    )

    return outstr


def generate_observation_summary_for_beamcut(parm_dict):
    xdt = parm_dict["xdt_data"]
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    obs_sum = xdt.attrs["summary"]

    tab_size = parm_dict["tab_size"]
    tab_count = 1
    header = f"{antenna}, {ddi}"
    outstr = make_header(header, "#", 60, 3)
    spc = " "

    outstr += (
        format_observation_summary(
            obs_sum,
            tab_size,
            tab_count,
            az_el_key=parm_dict["az_el_key"],
            phase_center_unit=parm_dict["phase_center_unit"],
            az_el_unit=parm_dict["az_el_unit"],
            time_format=parm_dict["time_format"],
        )
        + "\n"
    )
    for cut in xdt.children.values():
        outstr += f"{tab_count*tab_size*spc}{cut.name}:\n"
        outstr += f'{(tab_count+1)*tab_size*spc}{cut.attrs["direction"]} at {cut.attrs["time_string"]} UTC\n\n'

    return outstr
