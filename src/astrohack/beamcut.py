import pathlib
import toolviper.utils.logger as logger
import json

from astrohack.core.beamcut import process_beamcut_chunk
from astrohack.utils import get_default_file_name, add_caller_and_version_to_dict
from astrohack.utils.file import overwrite_file, check_if_file_can_be_opened
from astrohack.utils.graph import compute_graph
from astrohack.beamcut_mds import AstrohackBeamcutFile
import xarray as xr

from typing import Union, List


def beamcut(
    holog_name: str,
    beamcut_name: str = None,
    ant: Union[str, List[str]] = "all",
    ddi: Union[int, List[str]] = "all",
    correlations: str = "all",
    destination: str = None,
    lm_unit: str = "amin",
    azel_unit: str = "deg",
    dpi: int = 300,
    display: bool = False,
    y_scale: str = None,
    parallel: bool = False,
    overwrite: bool = False,
):

    check_if_file_can_be_opened(holog_name, "0.9.4")

    if beamcut_name is None:
        beamcut_name = get_default_file_name(
            input_file=holog_name, output_type=".beamcut.zarr"
        )

    if destination is not None:
        pathlib.Path(destination).mkdir(exist_ok=True)

    beamcut_params = locals()

    input_params = beamcut_params.copy()
    assert pathlib.Path(beamcut_params["holog_name"]).exists() is True, logger.error(
        f"File {beamcut_params['holog_name']} does not exists."
    )

    json_data = "/".join((beamcut_params["holog_name"], ".holog_json"))

    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)

    overwrite_file(beamcut_params["beamcut_name"], beamcut_params["overwrite"])

    executed_graph, graph_results = compute_graph(
        holog_json,
        process_beamcut_chunk,
        beamcut_params,
        ["ant", "ddi"],
        parallel=parallel,
        fetch_returns=True,
    )

    if executed_graph:
        logger.info("Finished processing")
        output_attr_file = "{name}/{ext}".format(
            name=beamcut_params["beamcut_name"], ext=".beamcut_input"
        )
        root = xr.DataTree(name="root")
        root.attrs.update(beamcut_params)
        add_caller_and_version_to_dict(root.attrs, direct_call=True)

        for xdtree in graph_results:
            ant, ddi = xdtree.name.split("-")
            if ant in root.keys():
                ant = root.children[ant].update({ddi: xdtree})
            else:
                ant_tree = xr.DataTree(name=ant, children={ddi: xdtree})
                root = root.assign({ant: ant_tree})

        root.to_zarr(beamcut_params["beamcut_name"], mode="w", consolidated=True)

        beamcut_mds = AstrohackBeamcutFile(beamcut_params["beamcut_name"])
        beamcut_mds.open()
        return beamcut_mds
    else:
        logger.warning("No data to process")
        return None
