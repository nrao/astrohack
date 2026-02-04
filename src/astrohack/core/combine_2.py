from copy import deepcopy

import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
from imageio.config.plugins import summary

from astrohack.utils import create_dataset_label
from astrohack.utils.file import load_image_xds
from scipy.interpolate import griddata
from astrohack.utils.constants import clight
from astrohack.utils.text import param_to_list


def process_combine_chunk(combine_chunk_params, output_mds):
    """
    Process a combine chunk
    Args:
        combine_chunk_params: Param dictionary for combine chunk
        output_mds: output mds file that contains combined data
    """

    ant_key = combine_chunk_params["this_ant"]
    ant_xdt = combine_chunk_params["xdt_data"]
    user_ddi_sel = combine_chunk_params["ddi"]
    ddi_list = param_to_list(user_ddi_sel, ant_xdt, "ddi")
    dataset_label = create_dataset_label(ant_key, None)

    nddi = len(ddi_list)
    if nddi == 0:
        logger.warning(f"Nothing to process for {ant_key}")
        return
    elif nddi == 1:
        ddi_key = ddi_list[0]

        if ddi_key in list(ant_xdt.keys()):
            logger.info(
                f"{dataset_label} has a single ddi to be combined, data copied from input file"
            )

            # Dataset already has the propper name!
            output_mds.add_node_to_tree(
                ant_xdt[ddi_key],
                dump_to_disk=True,
                running_in_parallel=combine_chunk_params["parallel"],
            )
        else:
            logger.warning(
                f"{dataset_label} has no {ddi_key}, nothing to process for this antenna"
            )
            return
    else:
        ddi_in_xdt_list = list(ant_xdt.keys())
        ddi_present_list = [ddi_key in ddi_in_xdt_list for ddi_key in ddi_list]
        if np.sum(ddi_present_list) == 0:
            logger.warning(
                f"{dataset_label} has no valid DDI in user selection (ddi = {user_ddi_sel})"
            )
            return
        ddi_ref_key = ddi_list[ddi_present_list.index(True)]

        out_xds = deepcopy(ant_xdt[ddi_ref_key].dataset)
        nddi = len(ddi_list)
        shape = list(out_xds["CORRECTED_PHASE"].values.shape)
        if out_xds.sizes["chan"] != 1:
            msg = f"Only single channel holographies supported"
            logger.error(msg)
            raise Exception(msg)
        npol = shape[2]
        npoints = shape[3] * shape[4]
        amp_sum = np.zeros((npol, npoints))
        pha_sum = np.zeros((npol, npoints))

        u, v = np.meshgrid(out_xds.u_prime.values, out_xds.v_prime.values)
        dest_u_axis = u.ravel()
        dest_v_axis = v.ravel()
        summary_list = []
        for i_ddi, ddi_key in enumerate(ddi_list):
            this_dataset_label = create_dataset_label(ant_key, ddi_key)
            if not ddi_present_list[i_ddi]:
                logger.warning(
                    f"{this_dataset_label} does not exist in input mds, skipping"
                )
                continue

            logger.info(f"Regridding {this_dataset_label}")
            this_xds = ant_xdt[ddi_key].dataset
            summary_list.append(this_xds.attrs["summary"])
            u, v = np.meshgrid(
                this_xds.u_prime.values,
                this_xds.v_prime.values,
            )
            loca_u_axis = u.ravel()
            loca_v_axis = v.ravel()
            for ipol in range(npol):
                thispha = this_xds["CORRECTED_PHASE"].values[0, 0, ipol, :, :].ravel()
                thisamp = this_xds["AMPLITUDE"].values[0, 0, ipol, :, :].ravel()
                repha = griddata(
                    (loca_u_axis, loca_v_axis),
                    thispha,
                    (dest_u_axis, dest_v_axis),
                    method="linear",
                )
                reamp = griddata(
                    (loca_u_axis, loca_v_axis),
                    thisamp,
                    (dest_u_axis, dest_v_axis),
                    method="linear",
                )
                amp_sum[ipol, :] += reamp
                if combine_chunk_params["weighted"]:
                    pha_sum[ipol, :] += repha * reamp
                else:
                    pha_sum[ipol, :] += repha

        if combine_chunk_params["weighted"]:
            phase = pha_sum / amp_sum
        else:
            phase = pha_sum / nddi
        amplitude = amp_sum / nddi

        out_xds["AMPLITUDE"] = xr.DataArray(
            amplitude.reshape(shape), dims=["time", "chan", "pol", "u_prime", "v_prime"]
        )
        out_xds["CORRECTED_PHASE"] = xr.DataArray(
            phase.reshape(shape), dims=["time", "chan", "pol", "u_prime", "v_prime"]
        )

        out_dataset_name = f"{ant_key}-ddi_99"
        output_mds.add_node_to_tree(
            xr.DataTree(name=out_dataset_name, dataset=out_xds),
            dump_to_disk=True,
            running_in_parallel=combine_chunk_params["parallel"],
        )
