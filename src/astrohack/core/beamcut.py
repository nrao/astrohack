import toolviper.utils.logger as logger
import numpy as np

from astrohack import get_proper_telescope
from astrohack.utils.file import load_holog_file
from astrohack.utils import create_dataset_label, data_statistics, statistics_to_text
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

    summary = this_xds.attrs["summary"]
    telescope = get_proper_telescope(
        summary["general"]["telescope name"], summary["general"]["antenna name"]
    )

    lm_offsets = this_xds.DIRECTIONAL_COSINES.values
    lm_deltas = np.diff(lm_offsets, axis=0)
    lm_angle = np.arctan2(lm_deltas[:, 1], lm_deltas[:, 0])

    lm_exclusion = sigma_clip_deltas(lm_deltas)
    print(lm_exclusion.shape)
    lm_deltas = lm_deltas[lm_exclusion, :]
    lm_angle = lm_angle[lm_exclusion]
    print(lm_deltas.shape, lm_angle.shape)

    timesteps =  np.arange(lm_angle.shape[0])
    timefracs =  np.arange(lm_offsets.shape[0])
    fig, ax = create_figure_and_axes(None, [2, 3])
    scatter_plot(ax[0, 0], timesteps, 'time intervals', lm_angle, 'LM angle [rad]')
    scatter_plot(ax[0, 1], timefracs, 'time intervals', lm_offsets[:, 0], 'L [rad]')
    scatter_plot(ax[0, 2], timefracs, 'time intervals', lm_offsets[:, 1], 'M [rad]')
    scatter_plot(ax[1, 1], timesteps, 'time intervals', lm_deltas[:, 0], 'delta L [rad]')
    scatter_plot(ax[1, 2], timesteps, 'time intervals', lm_deltas[:, 1], 'delta M [rad]')

    close_figure(fig, 'LM study', 'lm_simple.png', 300, False)

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




