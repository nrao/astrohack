from typing import Union, List

from astrohack.core.extract_locit import extract_spectral_info
from astrohack.utils.file import overwrite_file
from astrohack.utils.text import get_default_file_name
from astrohack.io.position_mds import AstrohackPositionFile


def fringefit_locit(
    fringefit_caltable: str,
    position_name: str,
    elevation_limit: float,
    polarization: str = "both",
    fit_engine: str = "scipy",
    fit_kterm: bool = False,
    fit_delay_rate: bool = False,
    ant: Union[str, List[str]] = "all",
    ddi: Union[str, int, List[int]] = "all",
    parallel: bool = True,
    overwrite: bool = False,
):
    position_name = get_default_file_name(
        fringefit_caltable, ".position.zarr", position_name
    )

    locit_params = locals()

    input_params = locit_params.copy()
    attributes = locit_params.copy()

    overwrite_file(locit_params["position_name"], locit_params["overwrite"])

    position_mds = AstrohackPositionFile.create_from_input_parameters(
        locit_params["position_name"], locit_params
    )

    ddi_dict = extract_spectral_info(locit_params)

    print(ddi_dict)

    position_mds.root.attrs.update(
        {
            "combined": True,
            # "telescope_name": locit_mds.root.attrs["telescope_name"],
            # "reference_antenna": locit_mds.root.attrs["reference_antenna"],
        }
    )

    return
