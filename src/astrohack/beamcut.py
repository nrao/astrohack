import pathlib

from toolviper.utils.parameter import validate

from astrohack.core.beamcut import process_beamcut_chunk
from astrohack.utils import get_default_file_name
from astrohack.utils.file import overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.io.beamcut_mds import AstrohackBeamcutFile
from astrohack.io.dio import open_holog
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

    if beamcut_name is None:
        beamcut_name = get_default_file_name(
            input_file=holog_name, output_type=".beamcut.zarr"
        )

    beamcut_params = locals()

    if destination is not None:
        pathlib.Path(destination).mkdir(exist_ok=True)

    overwrite_file(beamcut_name, overwrite)

    beamcut_mds = AstrohackBeamcutFile.create_from_input_parameters(
        beamcut_params["beamcut_name"], beamcut_params
    )

    holog_mds = open_holog(holog_name)
    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=holog_mds,
        chunk_function=process_beamcut_chunk,
        param_dict=beamcut_params,
        key_order=["ant", "ddi"],
        output_mds=beamcut_mds,
        parallel=parallel,
    )
    if executed_graph:
        beamcut_mds.write(mode="a")
        return beamcut_mds
    else:
        return None
