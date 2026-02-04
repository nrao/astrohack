from typing import Union, List

import toolviper.utils.parameter
import toolviper.utils.logger as logger

from astrohack import open_image
from astrohack.core.combine_2 import process_combine_chunk
from astrohack.utils.graph import compute_graph_to_mds_tree
from astrohack.utils.file import overwrite_file
from astrohack.utils.text import get_default_file_name
from astrohack.io.image_mds import AstrohackImageFile


# @toolviper.utils.parameter.validate()
def combine(
    image_name: str,
    combine_name: str = None,
    ant: Union[str, List[str]] = "all",
    ddi: Union[int, List[int], str] = "all",
    weighted: bool = False,
    parallel: bool = False,
    overwrite: bool = False,
) -> Union[AstrohackImageFile, None]:
    """Combine DDIs in a Holography image to increase SNR

    :param image_name: Input holography data file name. Accepted data format is the output from ``astrohack.holog.holog``
    :type image_name: str

    :param combine_name: Name of output file; File name will be appended with suffix *.combine.zarr*. Defaults to \
    *basename* of input file plus holography panel file suffix.
    :type combine_name: str, optional

    :param ant: List of antennas to be processed. None will use all antennas. Defaults to None, ex. ea25.
    :type ant: list or str, optional

    :param ddi: List of DDIs to be combined. None will use all DDIs. Defaults to None, ex. [0, ..., 8].
    :type ddi: list of int, optional

    :param weighted: Weight phases by the corresponding amplitudes.
    :type weighted: bool, optional

    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional

    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

    :return: Holography image object.
    :rtype: AstrohackImageFile

    .. _Description:
    **combine**

    Combine combines the amplitude and corrected_phase members of the selected DDIs in the input image file. Currently, \
    combine only supports the combination of these two quantities to avoid long regridding times. Hence, the output \
    image file (.combine.zarr file name) contains the combined amplitude and corrected_phase images, but the aperture \
    and beam images present in this file will be those present in the DDI with the lowest frequency.

    **AstrohackImageFile**

    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, \
    `ant` -> `ddi`. The image object produced by combine is special because it will always contain a single DDI.\
    The image object also provides a `summary()` helper function to list available keys for each file.\
     An outline of the image object structure when produced by combine is show below:

    .. parsed-literal::
        image_mds =
            {
            ant_0:{
                ddi_n: image_ds,
            },
            ⋮
            ant_n: …
        }

    **Example Usage**

    .. parsed-literal::
        from astrohack.combine import combine

        combine(
            "astrohack_obs.image.zarr",
            ant = "ea25"
            weight = False
        )
    """

    if combine_name is None:
        combine_name = get_default_file_name(
            input_file=image_name, output_type=".combine.zarr"
        )

    combine_params = locals()

    overwrite_file(combine_params["combine_name"], combine_params["overwrite"])

    image_mds = open_image(image_name)

    combine_mds = AstrohackImageFile.create_from_input_parameters(
        combine_name, combine_params
    )

    executed_graph = compute_graph_to_mds_tree(
        image_mds,
        process_combine_chunk,
        combine_params,
        ["ant"],
        combine_mds,
    )

    if executed_graph:
        combine_mds.write(mode="a")
        return combine_mds
    else:
        return None
