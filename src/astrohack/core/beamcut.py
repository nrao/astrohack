import toolviper.utils.logger as logger
import numpy as np
from scipy.stats import linregress

from astrohack import get_proper_telescope
from astrohack.utils.file import load_holog_file
from astrohack.utils import create_dataset_label, data_statistics, statistics_to_text, convert_unit
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
            cut_dict[f'{parallel_hand}_weight'] = np.angle(avg_vis[:, icorr])

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
                     'RR visibilities [Jy]')
        title += f'{cut["scan_number"]}, '
    title = title[:-2]
    close_figure(fig, title, 'lm_simple.png', 300, False)





