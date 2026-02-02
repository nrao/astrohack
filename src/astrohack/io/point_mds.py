import numpy as np
import pathlib

from typing import Union, List, Tuple

from astrohack.visualization.diagnostics import plot_array_configuration

from astrohack.core.extract_pointing_2 import (
    plot_pointing_in_time_together,
    plot_pointing_in_time_separately,
)
from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackPointFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackPointFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPointFile object
        :rtype: AstrohackPointFile
        """
        super().__init__(file=file)

    def plot_pointing_in_time(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        pointing_key: str = "TARGET",
        plot_antennas_separately: bool = False,
        azel_unit: str = "deg",
        time_unit: str = "hour",
        az_scale: Union[Tuple, List[float], np.array] = None,
        el_scale: Union[Tuple, List[float], np.array] = None,
        time_scale: Union[Tuple, List[float], np.array] = None,
        figure_size: Union[Tuple, List[float], np.array] = (5.0, 6.4),
        display: bool = False,
        dpi: int = 300,
    ):

        pathlib.Path(destination).mkdir(exist_ok=True)
        input_params = locals()

        if plot_antennas_separately:
            plot_pointing_in_time_separately(input_params, self)
        else:
            plot_pointing_in_time_together(input_params, self)
        return

    def plot_array_configuration(
        self,
        destination: str,
        stations: bool = True,
        zoff: bool = False,
        unit: str = "m",
        box_size: Union[int, float] = 5000,
        figure_size: Union[Tuple, List[float], np.array] = None,
        display: bool = False,
        dpi: int = 300,
    ):

        pathlib.Path(destination).mkdir(exist_ok=True)
        input_params = locals()
        plot_array_configuration(input_params, self.root, "point")
