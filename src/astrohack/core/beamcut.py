import toolviper.utils.logger as logger
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import astropy

from astrohack import get_proper_telescope
from astrohack.utils.file import load_holog_file
from astrohack.utils import create_dataset_label, data_statistics, statistics_to_text, convert_unit, sig_2_fwhm, \
    format_frequency, format_value_unit, to_db, fontsize
from astrohack.visualization import create_figure_and_axes, scatter_plot, close_figure
from astrohack.visualization.plot_tools import set_y_axis_lims_from_default
import matplotlib.ticker as mticker


def process_beamcut_chunk(beamcut_chunk_params):
    ddi = beamcut_chunk_params["this_ddi"]
    antenna = beamcut_chunk_params["this_ant"]

    _, ant_data_dict = load_holog_file(
        beamcut_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=antenna,
        ddi_id=ddi,
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

    beamcut_chunk_params['lm_unit'] = 'amin'
    beamcut_chunk_params['azel_unit'] = 'deg'
    beamcut_chunk_params['dpi'] = 300
    beamcut_chunk_params['display'] = False
    beamcut_chunk_params['y_scale'] = None

    plot_cuts_in_amplitude(cut_list, summary, beamcut_chunk_params)

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

        lm_angle, lm_dist, direction, xlabel = cut_direction_determination_and_label_creation(this_lm_offsets)
        hands_dict = get_parallel_hand_indexes(corr_axis)

        avg_vis = np.average(visibilities[time_selection, fchan:lchan, :], axis=1,
                             weights=weights[time_selection, fchan:lchan, :])
        avg_wei = np.average(weights[time_selection, fchan:lchan, :], axis=1)

        avg_time = np.average(time)*convert_unit('sec', 'day', 'time')
        timestr = astropy.time.Time(avg_time, format='mjd').to_value('iso', subfmt='date_hm')

        cut_dict = {
            'scan_number': scan_number,
            'time': time,
            'lm_offsets': this_lm_offsets,
            'lm_angle': lm_angle,
            'lm_dist': lm_dist,
            'available_corrs': hands_dict['parallel_hands'],
            'direction': direction,
            'xlabel': xlabel,
            'time_string': timestr
        }
        all_corr_ymax = 1e-34
        for parallel_hand in hands_dict['parallel_hands']:
            icorr = hands_dict[parallel_hand]
            amp = np.abs(avg_vis[:, icorr])
            maxamp = np.max(amp)
            if maxamp > all_corr_ymax:
                all_corr_ymax = maxamp
            cut_dict[f'{parallel_hand}_amplitude'] = amp
            cut_dict[f'{parallel_hand}_phase'] = np.angle(avg_vis[:, icorr])
            cut_dict[f'{parallel_hand}_weight'] = np.angle(avg_wei[:, icorr])
        cut_dict['max_amp'] = all_corr_ymax
        cut_list.append(cut_dict)

    return cut_list


def cut_direction_determination_and_label_creation(lm_offsets, angle_unit='deg'):
    dx = lm_offsets[-1, 0] - lm_offsets[0, 0]
    dy = lm_offsets[-1, 1] - lm_offsets[0, 1]
    lm_dist = np.sqrt(lm_offsets[:, 0] ** 2 + lm_offsets[:, 1] ** 2)
    imin_lm = np.argmin(lm_dist)
    lm_dist[:imin_lm] = -lm_dist[:imin_lm]

    if np.isclose(dx, dy, rtol=3e-1): # X case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi/2
        direction = 'mixed cut('
        if dy < 0 and dx < 0:
            direction += 'NW -> SE'

        elif dy < 0 < dx:
            direction += 'NE -> SW'

        elif dy > 0 > dx:
            direction += 'SW -> NE'

        else:
            direction += 'SE -> NW'

        direction += (r', $\theta$ = ' +
                      f'{format_value_unit(convert_unit('rad', angle_unit, 'trigonometric')*lm_angle, angle_unit)}')
        xlabel = 'Mixed offset'
    elif np.abs(dy) > np.abs(dx): # Elevation case
        result = linregress(lm_offsets[:, 1], lm_offsets[:, 0])
        lm_angle = np.arctan(result.slope)
        direction = 'El. cut ('
        if dy < 0:
            direction += 'N -> S'
            lm_dist *= -1 # Flip as sense is negative
        else:
            direction += 'S -> N'
        xlabel = 'Elevation offset'
    else: # Azimuth case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi/2
        direction = 'Az. cut ('
        if dx > 0:
            direction += 'E -> W'
        else:
            direction += 'W -> E'
            lm_dist *= -1 # Flip as sense is negative
        xlabel = 'Azimuth offset'

    direction += ')'

    return lm_angle, lm_dist, direction, xlabel


def get_parallel_hand_indexes(corr_axis):
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


def add_secondary_beam_hpbw_x_axis_to_plot(pb_fwhm, ax):
    sec_x_axis = ax.secondary_xaxis('top', functions=(lambda x: x*1.0, lambda xb: 1*xb))
    sec_x_axis.set_xlabel('Offset in Primary Beam HPBWs\n')
    sec_x_axis.set_xticks([])
    y_min, y_max = ax.get_ylim()
    x_lims = ax.get_xlim()
    pb_min, pb_max = np.ceil(x_lims/pb_fwhm)

    for itk in np.arange(pb_min, pb_max, 1):
        ax.axvline(itk*pb_fwhm, color='k', linestyle='--', linewidth=0.5)
        ax.text(itk*pb_fwhm, y_max, f'{itk+1}', va='bottom', ha='center')


def add_lobe_identification_to_plot(ax, centers, peaks, y_off, attenunation_plot=False):
    if attenunation_plot:
        plot_peaks = to_db(peaks/np.max(peaks)) # maximum of peaks is always the PB
    else:
        plot_peaks = peaks

    for i_peak, peak in enumerate(plot_peaks):
        ax.text(centers[i_peak], peak+y_off, f'{i_peak+1})', ha='center', va='bottom')


def make_parallel_hand_subplot_title(direction, time_string):
    return f'{direction}, {time_string} UTC'


def plot_single_cut_parallel_corrs(cut_dict, axes, par_dict):
    # Init
    sub_title = make_parallel_hand_subplot_title(cut_dict["direction"], cut_dict["time_string"])
    max_amp = cut_dict['max_amp']
    y_off = 0.05*max_amp
    lm_unit = par_dict['lm_unit']
    lm_fac = convert_unit('rad', lm_unit, 'trigonometric')

    # Loop over correlations
    for i_corr, parallel_hand in enumerate(cut_dict['available_corrs']):
        # Init labels
        this_ax = axes[i_corr]
        x_data = lm_fac * cut_dict['lm_dist']
        y_data = cut_dict[f'{parallel_hand}_amplitude']
        fit_data = cut_dict[f'{parallel_hand}_amp_fit']
        xlabel = f'{cut_dict['xlabel']} [{lm_unit}]'
        ylabel = f'{parallel_hand} Amplitude [ ]'

        # Call plotting tool
        scatter_plot(this_ax, x_data, xlabel, y_data, ylabel, model=fit_data,  model_marker='', title=sub_title,
                     data_marker='+', residuals_marker='.', model_linestyle='-', data_label=f'{parallel_hand} data',
                     model_label=f'{parallel_hand} fit', data_color='red', model_color='blue', residuals_color='black',)

        # Add fit peak identifiers
        add_lobe_identification_to_plot(this_ax, lm_fac * cut_dict[f'{parallel_hand}_amp_fit_pars'][0::3],
                                        cut_dict[f'{parallel_hand}_amp_fit_pars'][1::3], y_off, attenunation_plot=False)

        # equalize Y scale between correlations
        set_y_axis_lims_from_default(this_ax, par_dict['y_scale'], (-y_off, max_amp+3*y_off))

        add_secondary_beam_hpbw_x_axis_to_plot(cut_dict[f'{parallel_hand}_pb_fwhm']*lm_fac, this_ax)

        # Add bounded box with Beam parameters
        add_beam_parameters_box(this_ax, cut_dict[f'{parallel_hand}_pb_center']*lm_fac,
                                cut_dict[f'{parallel_hand}_pb_fwhm']*lm_fac,
                                cut_dict[f'{parallel_hand}_first_side_lobe_ratio'],
                                lm_unit)


def add_beam_parameters_box(ax, pb_center, pb_fwhm, sidelobe_ratio, lm_unit, alpha=0.8, x_pos=0.05, y_pos=0.95):
    pars_str = f'PB off. = {format_value_unit(pb_center, lm_unit, 3)}\n'
    pars_str += f'PB FWHM = {format_value_unit(pb_fwhm, lm_unit, 3)}\n'
    pars_str += f'FSLR = {format_value_unit(to_db(sidelobe_ratio), 'dB', 2)}'
    bounds_box = dict(boxstyle='square', facecolor='white', alpha=alpha)
    ax.text(0.05, 0.95, pars_str, transform=ax.transAxes, verticalalignment='top',
                     bbox=bounds_box)


def create_beamcut_header(summary, par_dict):
    azel_unit = par_dict['azel_unit']
    antenna = par_dict['this_ant']
    ddi = par_dict['this_ddi']
    freq_str = format_frequency(summary['spectral']['rep. frequency'], decimal_places=3)
    raw_azel = np.array(summary['general']["az el info"]["mean"])
    mean_azel = convert_unit('rad', azel_unit, 'trigonometric') * raw_azel
    title = f'Beam cut for {create_dataset_label(antenna, ddi, separator=',')}, ' + r'$\nu$ = ' + f'{freq_str}, '
    title += f'Az ~ {format_value_unit(mean_azel[0], 'deg', decimal_places=0)}, '
    title += f'El ~ {format_value_unit(mean_azel[1], 'deg', decimal_places=0)}'
    return title

def plot_cuts_in_amplitude(cut_list, summary, par_dict):
    # Init
    n_cuts = len(cut_list)
    antenna = par_dict['this_ant']
    ddi = par_dict['this_ddi']

    # Loop over cuts
    fig, axes = create_figure_and_axes([12, 1+n_cuts*4], [n_cuts, 2])
    for icut, cut_dict in enumerate(cut_list):
        plot_single_cut_parallel_corrs(cut_dict, axes[icut, :], par_dict)

    # Header creation
    title = create_beamcut_header(summary, par_dict)

    filename = f'beamcut_{antenna}_{ddi}.png'
    close_figure(fig, title, filename, par_dict['dpi'], par_dict['display'])


def gaussian(x_axis, x_off, amp, fwhm):
    sigma = fwhm / sig_2_fwhm
    return amp * np.exp(-(x_axis - x_off)**2/(2*sigma**2))


def build_multi_gaussian_initial_guesses(x_data, y_data, pb_fwhm, min_dist_fraction=1.3):
    p0 = []
    step = float(np.median(np.diff(x_data)))
    min_dist = np.abs(min_dist_fraction * pb_fwhm / step)
    peaks, _ = find_peaks(y_data, distance=min_dist)
    dx = x_data[-1] - x_data[0]
    if dx < 0:
        peaks = peaks[::-1]
    for ipeak in peaks:
        p0.extend([x_data[ipeak], y_data[ipeak], pb_fwhm])
    return p0, len(peaks)

def multi_gaussian(xdata, *args):
    nargs = len(args)
    if nargs%3 != 0:
        raise ValueError('Number of arguments should be multiple of 3')
    y_values = np.zeros_like(xdata)
    for iarg in range(0, nargs, 3):
        y_values += gaussian(xdata, args[iarg], args[iarg+1], args[iarg+2])
    return y_values


def beamcut_fit(cut_list, telescope, summary):
    wavelength = summary["spectral"]["rep. wavelength"]
    primary_fwhm = 1.2 * wavelength / telescope.diameter

    for cut_dict in cut_list:
        x_data = cut_dict['lm_dist']
        for parallel_hand in cut_dict['available_corrs']:
            y_data = cut_dict[f'{parallel_hand}_amplitude']
            p0, n_peaks = build_multi_gaussian_initial_guesses(x_data, y_data, primary_fwhm)
            results = curve_fit(multi_gaussian, x_data, y_data, p0=p0)
            fit_pars = results[0]
            fit = multi_gaussian(x_data, *fit_pars)

            cut_dict[f'{parallel_hand}_amp_fit_pars'] = fit_pars
            cut_dict[f'{parallel_hand}_amp_fit'] = fit
            cut_dict[f'{parallel_hand}_n_peaks'] = n_peaks

            centers = fit_pars[0::3]
            amps = fit_pars[1::3]
            sigmas = fit_pars[2::3]

            i_pb = np.argmax(amps)
            cut_dict[f'{parallel_hand}_pb_fwhm'] = sigmas[i_pb]
            cut_dict[f'{parallel_hand}_pb_center'] = centers[i_pb]

            left_first_sl_amp = amps[i_pb-1]
            right_first_sl_amp = amps[i_pb+1]
            cut_dict[f'{parallel_hand}_first_side_lobe_ratio'] = left_first_sl_amp / right_first_sl_amp

    return cut_list






