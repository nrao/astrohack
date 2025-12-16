import toolviper.utils.logger as logger
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from astrohack import get_proper_telescope
from astrohack.utils.file import load_holog_file
from astrohack.utils import create_dataset_label, data_statistics, statistics_to_text, convert_unit, sig_2_fwhm
from astrohack.visualization import create_figure_and_axes, scatter_plot, close_figure


def process_beamcut_chunk(beamcut_chunk_params):
    ddi = beamcut_chunk_params["this_ddi"]
    antenna = beamcut_chunk_params["this_ant"]

    _, ant_data_dict = load_holog_file(
        beamcut_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=beamcut_chunk_params["this_ant"],
        ddi_id=beamcut_chunk_params["this_ddi"],
    )
    # This assumes that there will be no more than one mapping
    this_xds = ant_data_dict[ddi]['map_0']
    logger.info(f"processing {create_dataset_label(antenna, ddi)}")

    print(this_xds)

    scan_time_ranges = this_xds.attrs['scan_time_ranges']
    scan_list = this_xds.attrs['scan_list']
    summary = this_xds.attrs["summary"]

    telescope = get_proper_telescope(
        summary["general"]["telescope name"], summary["general"]["antenna name"]
    )

    lm_offsets = this_xds.DIRECTIONAL_COSINES.values
    time_axis = this_xds.time.values
    corr_axis = this_xds.pol.values
    visibilities = this_xds.VIS.values
    weights = this_xds.WEIGHT.values

    cut_list = extract_cuts_from_visibilities(scan_list, scan_time_ranges, time_axis, corr_axis, lm_offsets,
                                              visibilities, weights)

    beamcut_fit(cut_list, telescope, summary)
    plot_cuts(cut_list)
    # lm_deltas = np.diff(lm_offsets, axis=0)
    # lm_angle = np.arctan2(lm_deltas[:, 1], lm_deltas[:, 0])
    #
    # lm_exclusion = sigma_clip_deltas(lm_deltas)
    # print(lm_exclusion.shape)
    # lm_deltas = lm_deltas[lm_exclusion, :]
    # lm_angle = lm_angle[lm_exclusion]
    # print(lm_deltas.shape, lm_angle.shape)
    #
    # timesteps =  np.arange(lm_angle.shape[0])
    # timefracs =  np.arange(lm_offsets.shape[0])
    # fig, ax = create_figure_and_axes(None, [2, 3])
    # scatter_plot(ax[0, 0], timesteps, 'time intervals', lm_angle, 'LM angle [rad]')
    # scatter_plot(ax[0, 1], timefracs, 'time intervals', lm_offsets[:, 0], 'L [rad]')
    # scatter_plot(ax[0, 2], timefracs, 'time intervals', lm_offsets[:, 1], 'M [rad]')
    # scatter_plot(ax[1, 1], timesteps, 'time intervals', lm_deltas[:, 0], 'delta L [rad]')
    # scatter_plot(ax[1, 2], timesteps, 'time intervals', lm_deltas[:, 1], 'delta M [rad]')
    #
    # close_figure(fig, 'LM study', 'lm_simple.png', 300, False)

    # vis = this_xds.VIS.values

def sigma_clip_deltas(lm_deltas, clip=5):
    l_delta_stats = data_statistics(lm_deltas[:, 0])
    m_delta_stats = data_statistics(lm_deltas[:, 1])
    print('L before:\n\t',statistics_to_text(l_delta_stats, num_format='.6f'))
    print('M before:\n\t',statistics_to_text(m_delta_stats, num_format='.6f'))

    sigma_exclusion = np.logical_and(np.abs(lm_deltas[:, 0]) < clip * l_delta_stats['rms'],
                                     np.abs(lm_deltas[:, 1]) < clip * m_delta_stats['rms'])

    l_delta_stats = data_statistics(lm_deltas[sigma_exclusion, 0])
    m_delta_stats = data_statistics(lm_deltas[sigma_exclusion, 1])
    print('L after:\n\t',statistics_to_text(l_delta_stats, num_format='.6f'))
    print('M after:\n\t',statistics_to_text(m_delta_stats, num_format='.6f'))
    return sigma_exclusion


def time_scan_selection(scan_time_ranges, time_axis):
    time_selections = []
    for scan_time_range in scan_time_ranges:
        time_selection = np.logical_and(time_axis >= scan_time_range[0],
                                        time_axis < scan_time_range[1])
        time_selections.append(time_selection)
    return time_selections


def extract_cuts_from_visibilities(scan_list, scan_time_ranges, time_axis, corr_axis, lm_offsets,
                                   visibilities, weights):
    cut_list = []
    nchan = visibilities.shape[1]
    fchan = 4
    lchan = int(nchan - fchan)
    for iscan, scan_number in enumerate(scan_list):
        scan_time_range = scan_time_ranges[iscan]
        time_selection = np.logical_and(time_axis >= scan_time_range[0],
                                        time_axis < scan_time_range[1])
        time = time_axis[time_selection]
        this_lm_offsets = lm_offsets[time_selection, :]

        lm_angle, lm_dist = cut_direction_determination(this_lm_offsets)
        hands_dict = get_hand_indexes(corr_axis)

        avg_vis = np.average(visibilities[time_selection, fchan:lchan, :], axis=1,
                             weights=weights[time_selection, fchan:lchan, :])
        avg_wei = np.average(weights[time_selection, fchan:lchan, :], axis=1)

        cut_dict = {
            'scan_number': scan_number,
            'time': time,
            'lm_offsets': this_lm_offsets,
            'lm_angle': lm_angle,
            'lm_dist': lm_dist,
            'available_corrs': hands_dict['parallel_hands']
        }
        for parallel_hand in hands_dict['parallel_hands']:
            icorr = hands_dict[parallel_hand]
            cut_dict[f'{parallel_hand}_amplitude'] = np.abs(avg_vis[:, icorr])
            cut_dict[f'{parallel_hand}_phase'] = np.angle(avg_vis[:, icorr])
            cut_dict[f'{parallel_hand}_weight'] = np.angle(avg_wei[:, icorr])

        cut_list.append(cut_dict)

    return cut_list


def cut_direction_determination(lm_offsets):
    result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
    lm_angle = np.arctan(result.slope) - np.pi/2

    lm_dist = np.sqrt(lm_offsets[:, 0]**2 + lm_offsets[:, 1]**2)
    imin = np.argmin(lm_dist)
    lm_dist[:imin] = -lm_dist[:imin]
    return lm_angle, lm_dist


def get_hand_indexes(corr_axis):
    if 'L' in corr_axis[0] or 'R' in corr_axis[0]:
        parallel_hands = ['RR', 'LL']
    else:
        parallel_hands = ['XX', 'YY']

    hands_dict ={
        'parallel_hands': parallel_hands
    }
    for icorr, corr in enumerate(corr_axis):
        hands_dict[corr] = icorr
    return hands_dict


def plot_cuts(cut_list):
    n_cuts = len(cut_list)
    print(n_cuts)
    title = 'Scans: '
    fig, ax = create_figure_and_axes(None, [n_cuts, 3])
    for icut, cut in enumerate(cut_list):
        sub_title = (f'Cut angle w.r.t. North = '
                     f'{cut["lm_angle"] *  convert_unit('rad', 'deg', 'trigonometric'):.1f} deg')
        scatter_plot(ax[icut, 0], cut['lm_dist'], 'LM distance [rad]', cut['lm_offsets'][:, 0], 'L [rad]',
                     title=sub_title)
        scatter_plot(ax[icut, 1], cut['lm_dist'], 'LM distance [rad]', cut['lm_offsets'][:, 1], 'M [rad]')
        scatter_plot(ax[icut, 2], cut['lm_dist'], 'LM distance [rad]', cut['RR_amplitude'][:],
                     'RR visibilities [Jy]', model=cut['RR_amp_fit'][:])
        title += f'{cut["scan_number"]}, '
    title = title[:-2]
    close_figure(fig, title, 'lm_simple.png', 300, False)


def gaussian(x_axis, x_off, amp, fwhm):
    sigma = fwhm / sig_2_fwhm
    return amp * np.exp(-(x_axis - x_off)**2/(2*sigma**2))


def primary_and_first_side_lobes(x_axis, primary_fwhm, primary_amp, primary_center, first_sidelobe_offset,
                                 left_sidelobe_amp, left_sidelobe_fwhm, right_sidelobe_amp, right_sidelobe_fwhm):
    primary = gaussian(x_axis, primary_center, primary_fwhm, primary_amp)
    left_sidelobe = gaussian(x_axis, primary_center-first_sidelobe_offset,
                             left_sidelobe_amp, left_sidelobe_fwhm)
    right_sidelobe = gaussian(x_axis, primary_center+first_sidelobe_offset,
                              right_sidelobe_amp, right_sidelobe_fwhm)
    full_beam = primary + left_sidelobe + right_sidelobe
    return full_beam


def pb_fsl(x_axis, pb_off, pb_amp, pb_fwhm, lfsl_off, lfsl_amp, lfsl_fwhm, rfsl_off, rfsl_amp, rfsl_fwhm):
    pb = gaussian(x_axis, pb_off, pb_amp, pb_fwhm)
    lfsl = gaussian(x_axis, lfsl_off, lfsl_amp, lfsl_fwhm)
    rfsl = gaussian(x_axis, rfsl_off, rfsl_amp, rfsl_fwhm)
    return pb + lfsl + rfsl


def pb_fsl_ssl(x_axis, pb_off, pb_amp, pb_fwhm, lfsl_off, lfsl_amp, lfsl_fwhm, rfsl_off, rfsl_amp, rfsl_fwhm,
               lssl_off, lssl_amp, lssl_fwhm, rssl_off, rssl_amp, rssl_fwhm):
    lssl = gaussian(x_axis, lssl_off, lssl_amp, lssl_fwhm)
    rssl = gaussian(x_axis, rssl_off, rssl_amp, rssl_fwhm)
    pb_fsl_model = pb_fsl(x_axis, pb_off, pb_amp, pb_fwhm, lfsl_off, lfsl_amp, lfsl_fwhm, rfsl_off, rfsl_amp, rfsl_fwhm)
    return pb_fsl_model + lssl + rssl


def beamcut_fit(cut_list, telescope, summary):
    wavelength = summary["spectral"]["rep. wavelength"]
    primary_fwhm = 1.2 * wavelength / telescope.diameter
    for cut_dict in cut_list:
        x_data = cut_dict['lm_dist']
        step = np.median(np.diff(x_data))
        min_dist = 1.5 * primary_fwhm / step
        for parallel_hand in cut_dict['available_corrs']:
            y_data = cut_dict[f'{parallel_hand}_amplitude']
            # ymax = np.max(y_data)
            # p0 = [primary_fwhm, ymax, 0.0, 1*primary_fwhm, 0.2*ymax, primary_fwhm, 0.2*ymax, primary_fwhm]
            # results = curve_fit(primary_and_first_side_lobes, x_data, y_data, p0=p0)
            # fit_pars = results[0]
            # fit = primary_and_first_side_lobes(x_data, *fit_pars)
            # p0 = [0.0, ymax, primary_fwhm]
            # results = curve_fit(gaussian, x_data, y_data, p0=p0)
            # fit_pars = results[0]
            # fit = gaussian(x_data, *fit_pars)
            peaks, _ = find_peaks(y_data, distance=min_dist)
            n_peaks = len(peaks)
            if n_peaks == 1:
                fit_func = gaussian
                i_peak = peaks[0]
                p0 = [x_data[i_peak], y_data[i_peak], primary_fwhm]
            elif n_peaks == 3:
                fit_func = pb_fsl
                i_lfsl_peak = peaks[0]
                i_pb_peak = peaks[1]
                i_rfsl_peak = peaks[2]
                p0 = [x_data[i_pb_peak], y_data[i_pb_peak], primary_fwhm,
                      x_data[i_lfsl_peak], y_data[i_lfsl_peak], primary_fwhm,
                      x_data[i_rfsl_peak], y_data[i_rfsl_peak], primary_fwhm,]
            elif n_peaks == 5:
                fit_func = pb_fsl_ssl
                i_lssl_peak = peaks[0]
                i_lfsl_peak = peaks[1]
                i_pb_peak = peaks[2]
                i_rfsl_peak = peaks[3]
                i_rssl_peak = peaks[4]

                p0 = [x_data[i_pb_peak], y_data[i_pb_peak], primary_fwhm,
                      x_data[i_lfsl_peak], y_data[i_lfsl_peak], primary_fwhm,
                      x_data[i_rfsl_peak], y_data[i_rfsl_peak], primary_fwhm,
                      x_data[i_lssl_peak], y_data[i_lssl_peak], primary_fwhm,
                      x_data[i_rssl_peak], y_data[i_rssl_peak], primary_fwhm,
                      ]
            else:
                raise RuntimeError(f"Don't know how to fit a beam cut with {n_peaks} peaks")

            print(min_dist, primary_fwhm, n_peaks)
            results = curve_fit(fit_func, x_data, y_data, p0=p0)
            fit_pars = results[0]
            fit = fit_func(x_data, *fit_pars)

            print(fit_pars)
            cut_dict[f'{parallel_hand}_amp_fit_pars'] = fit_pars
            cut_dict[f'{parallel_hand}_amp_fit'] = fit

    return cut_list






