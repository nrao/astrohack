import pathlib
import json
import xarray as xr

import toolviper.utils.logger as logger

from toolviper.utils.parameter import validate

from astrohack.core.beamcut import process_beamcut_chunk
from astrohack.utils import get_default_file_name, add_caller_and_version_to_dict
from astrohack.utils.file import overwrite_file, check_if_file_can_be_opened
from astrohack.utils.graph import compute_graph
from astrohack.io.beamcut_mds import AstrohackBeamcutFile
from astrohack.utils.validation import custom_plots_checker

from typing import Union, List


@validate(custom_checker=custom_plots_checker)
def beamcut(
    holog_name: str,
    beamcut_name: str = None,
    ant: Union[str, List[str]] = "all",
    ddi: Union[int, List[str]] = "all",
    destination: str = None,
    lm_unit: str = "amin",
    azel_unit: str = "deg",
    dpi: int = 300,
    display: bool = False,
    y_scale: list[float] = None,
    parallel: bool = False,
    overwrite: bool = False,
):
    """
    Process beamcut data from a .holog.zarr file to produce reports and plots.

    :param holog_name: Name of the .holog.zarr file to use as input.
    :type holog_name: str

    :param beamcut_name: Name for the output .beamcut.zarr file to save data.
    :type beamcut_name: str

    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25.
    :type ant: list or str, optional

    :param ddi: List of ddi to be processed, defaults to "all" when None, ex. 0.
    :type ddi: list or int, optional

    :param destination: Destination directory for plots and reports if not None, defaults to None.
    :type destination: str, optional

    :param lm_unit: Unit for L/M offsets in plots and report, default is "amin".
    :type lm_unit: str, optional

    :param azel_unit: Unit for Az/El information in plots and report, default is "deg".
    :type azel_unit: str, optional

    :param dpi: Resolution in pixels, defaults to 300.
    :type dpi: int, optional

    :param display: Display plots during execution, defaults to False.
    :type display: bool, optional

    :param y_scale: Define amplitude plot Y scale, defaults to None.
    :type y_scale: str, optional

    :param parallel: Process beamcuts in parallel, defaults to False.
    :type parallel: bool, optional

    :param overwrite: Overwrite previously existing beamcut file of same name, defaults to False.
    :type overwrite: bool, optional

    :return: Beamcut mds object
    :rtype: AstrohackBeamcutFile

    .. _Description:
    **AstrohackBeamcutFile**

    The beamcut mds object allows the user to access the underlying xarray datatree using compound keys, which are in \
    order of depth, `ant` -> `ddi`. This object also provides a `summary()` method to list available data and available\
     data visualization methods.

      An outline of the beamcut mds data tree is show below:

    .. parsed-literal::
        image_mds =
            {
            ant_0:{
                ddi_0: {
                    cut_0: beamcut_ds
                    ⋮
                    cut_p: beamcut_ds
                },
                ddi_m: …
            },
            ⋮
            ant_n: …
        }

    **Example Usage**

    .. parsed-literal::
        from astrohack import beamcut

        beamcut(
            holog_name="astrohack_observation.holog.zarr",
            beamcut_name="astrohack_observation.beamcut.zarr",
            destination="beamcut_exports",
            display=False,
            ant='ea25',
            overwrite=True,
            parallel=True
        )
    """

    check_if_file_can_be_opened(holog_name, "0.9.5")

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
