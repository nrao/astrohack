import pathlib
import toolviper.utils.logger as logger
import json

from astrohack.core.beamcut import process_beamcut_chunk
from astrohack.utils import get_default_file_name
from astrohack.utils.file import overwrite_file
from astrohack.utils.graph import compute_graph
from astrohack.utils.data import write_meta_data

from typing import Union, List

def beamcut_tool(
        holog_name: str,
        beamcut_name: str = None,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[str]] = "all",
        correlations: str = "all",
        parallel: bool = False,
        overwrite: bool = False,
):

    if beamcut_name is None:
        beamcut_name = get_default_file_name(
            input_file=holog_name, output_type=".beamcut.zarr"
        )

    beamcut_params = locals()

    input_params = beamcut_params.copy()
    assert pathlib.Path(beamcut_params["holog_name"]).exists() is True, logger.error(
        f"File {beamcut_params['holog_name']} does not exists."
    )

    json_data = "/".join((beamcut_params["holog_name"], ".holog_json"))

    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)

    overwrite_file(beamcut_params["beamcut_name"], beamcut_params['overwrite'])

    if compute_graph(
         holog_json,
        process_beamcut_chunk,
        beamcut_params,
        ["ant", "ddi"],
        parallel=parallel,
        ):
        logger.info("Finished processing")
        output_attr_file = "{name}/{ext}".format(
            name=beamcut_params["beamcut_name"], ext=".beamcut_input"
        )
        # write_meta_data(output_attr_file, input_params)
        # beamcut_mds = AstrohackbeamcutFile(beamcut_params["beamcut_name"])
        # beamcut_mds.open()
        #
        # return beamcut_mds
        return None
    else:
        logger.warning("No data to process")
        return None