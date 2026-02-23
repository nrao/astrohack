import json
import toolviper

import numpy as np

from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import extract_holog
from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.io.dio import open_panel, open_holog

from astrohack.io.dio import open_image
from toolviper.utils import logger


def get_center_pixel(file, antenna, ddi):
    mds = open_image(file)[antenna][ddi]

    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]

    aperture_center_pixels = mds.APERTURE.values[
        ..., aperture_shape[0] // 2, aperture_shape[1] // 2
    ]
    beam_center_pixels = mds.BEAM.values[..., beam_shape[0] // 2, beam_shape[1] // 2]

    return {
        "aperture": np.squeeze(aperture_center_pixels),
        "beam": np.squeeze(beam_center_pixels),
    }


def get_grid_parameters(file, antenna, mapping, ddi):
    holog_mds = open_holog(file)
    xds = holog_mds[antenna][ddi][mapping]

    beam_summary = xds.attrs["summary"]["beam"]
    cell_size = beam_summary["cell size"]
    grid_size = beam_summary["grid size"]
    return cell_size, grid_size


def generate_verification_json(
    antenna, ddi, path="./data", write=True, generate_files=True
):
    from astrohack.io.dio import open_panel

    if generate_files:
        generate_verification_files()

    numerical_dict = {
        "vla": {
            "pixels": {
                "before": {"aperture": [], "beam": []},
                "after": {"aperture": [], "beam": []},
            },
            "cell_size": [],
            "grid_size": [],
            "offsets": [],
        }
    }

    for tag in ["before", "after"]:
        pixels = get_center_pixel(
            file=f"{path}/{tag}.split.image.zarr", antenna=antenna, ddi=ddi
        )

        numerical_dict["vla"]["pixels"][tag]["aperture"] = list(
            map(str, pixels["aperture"])
        )
        numerical_dict["vla"]["pixels"][tag]["beam"] = list(map(str, pixels["beam"]))

    cell_size, grid_size = get_grid_parameters(
        f"{path}/before.split.holog.zarr", antenna, "map_0", ddi
    )

    numerical_dict["vla"]["cell_size"] = [-cell_size, cell_size]
    numerical_dict["vla"]["grid_size"] = [grid_size[0], grid_size[1]]

    # Fill panel offsets
    panel_list = ["3-4", "5-27", "5-37", "5-38"]

    m_to_mils = 39370.1

    before_mds = open_panel(f"{path}/before.split.panel.zarr")
    after_mds = open_panel(f"{path}/after.split.panel.zarr")

    before_shift = (
        before_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * m_to_mils
    )
    after_shift = (
        after_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * m_to_mils
    )

    numerical_dict["vla"]["offsets"] = np.mean(
        after_shift - before_shift, axis=1
    ).tolist()
    print(np.mean(after_shift - before_shift, axis=1).tolist())

    if write:
        with open("./holog_numerical_verification.json", "w") as json_file:
            json.dump(numerical_dict, json_file)

    return numerical_dict


def generate_verification_files():
    for stub in ["before", "after"]:
        toolviper.utils.data.download(
            file=f"ea25_cal_small_{stub}_fixed.split.ms", folder="data/"
        )

        extract_pointing(
            ms_name=f"data/ea25_cal_small_{stub}_fixed.split.ms",
            point_name=f"data/{stub}.split.point.zarr",
            overwrite=True,
            parallel=False,
        )

        # Extract holography data using holog_obd_dict
        extract_holog(
            ms_name=f"data/ea25_cal_small_{stub}_fixed.split.ms",
            point_name=f"data/{stub}.split.point.zarr",
            holog_name=f"data/{stub}.split.holog.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True,
        )

        holog(
            holog_name=f"data/{stub}.split.holog.zarr",
            image_name=f"data/{stub}.split.image.zarr",
            overwrite=True,
            parallel=False,
        )

        panel(
            image_name=f"data/{stub}.split.image.zarr",
            panel_model="rigid",
            parallel=False,
            overwrite=True,
        )


def generate_panel_mask_array(generate_files=True):
    if generate_files:
        generate_verification_files()

    panel(
        image_name="data/before.split.image.zarr",
        clip_type="absolute",
        clip_level=0.0,
        parallel=False,
        overwrite=True,
    )

    before_mds = open_panel("data/before.split.panel.zarr")

    with open("data/panel_cutoff_mask.npy", "wb") as outfile:
        np.save(outfile, before_mds["ant_ea25"]["ddi_0"].MASK.values)
