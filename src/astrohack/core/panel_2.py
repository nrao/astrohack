import xarray as xr

import toolviper.utils.logger as logger

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.utils import create_dataset_label


def process_panel_chunk(panel_chunk_params, output_mds):
    """
    Process a chunk of the holographies, usually a chunk consists of an antenna over a ddi
    Args:
        panel_chunk_params: dictionary of inputs
        output_mds: Output panel mds object
    """

    clip_level = panel_chunk_params["clip_level"]
    ddi_key = panel_chunk_params["this_ddi"]
    ant_key = panel_chunk_params["this_ant"]
    inputxds = panel_chunk_params["xdt_data"].dataset
    dataset_label = create_dataset_label(ant_key, ddi_key)
    logger.info(f"processing {dataset_label}")
    if isinstance(clip_level, dict):
        ant_name = ant_key.split("_")[1]
        ddi_name = int(ddi_key.split("_")[1])
        try:
            clip_level = clip_level[ant_name][ddi_name]
        except KeyError:
            msg = f"{dataset_label} combination not found in clip_level dictionary"
            logger.error(msg)
            raise RuntimeError(msg)

    surface = AntennaSurface(
        inputxds,
        clip_type=panel_chunk_params["clip_type"],
        pol_state=panel_chunk_params["polarization_state"],
        clip_level=clip_level,
        pmodel=panel_chunk_params["panel_model"],
        panel_margins=panel_chunk_params["panel_margins"],
        patch_phase=False,
        use_detailed_mask=panel_chunk_params["use_detailed_mask"],
    )

    surface.fit_surface()
    surface.correct_surface()

    xds = surface.export_xds()
    dataset_name = f"{ant_key}-{ddi_key}"
    output_mds.add_node_to_tree(
        xr.DataTree(name=dataset_name, dataset=xds),
        dump_to_disk=True,
        running_in_parallel=panel_chunk_params["parallel"],
    )
