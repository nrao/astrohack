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
        pointing_key: str = "DIRECTIONAL_COSINES",
        plot_antennas_separately: bool = False,
        azel_unit: str = "deg",
        time_unit: str = "hour",
        az_scale: Union[Tuple, List[float], np.array] = None,
        el_scale: Union[Tuple, List[float], np.array] = None,
        time_scale: Union[Tuple, List[float], np.array] = None,
        figure_size: Union[Tuple, List[float], np.array] = (5.0, 6.4),
        display: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot Pointing for antennas in time.

        :param destination: Name of the destination folder to contain plot(s)
        :type destination: str

        :param ant: Antenna(s) to plot, default is "all"
        :type ant: str, list, optional

        :param pointing_key: Which xds pointing data key to plot, defaults to "DIRECTIONAL_COSINES"
        :type pointing_key: str, optional

        :param plot_antennas_separately: Create an individual plot file for each antenna?
        :type plot_antennas_separately: bool, optional

        :param azel_unit: Unit for Azimuth and Elevation in the plot(s), valid values are trigonometric units, default \
        is deg
        :type azel_unit: str, optional

        :param time_unit: Unit for time in the plot(s), valid values are time units, default is hour
        :type time_unit: str, optional

        :param az_scale: Azimuth plot limits, defaults to all Azimuths present when None.
        :type az_scale: Union[Tuple, List[float], np.array], optional

        :param el_scale: Elevation plot limits, defaults to all Elevations present when None.
        :type el_scale: Union[Tuple, List[float], np.array], optional

        :param time_scale: Time plot limits, defaults to all times present when None
        :type time_scale: Union[Tuple, List[float], np.array], optional

        :param display: Display plot(s) inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:
        Plot antenna pointing info in time together in one plot, or individually for each antenna.
        """

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
    ) -> None:
        """Plot antenna positions.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param stations: Add station names to the plot, defaults to True
        :type stations: bool, optional

        :param zoff: Add Elevation offsets to the plots, defaults to False
        :type zoff: bool, optional

        :param unit: Unit for the plot, valid values are length units, default is km
        :type unit: str, optional

        :param box_size: Size of the box for plotting the inner part of the array in unit, default is 5 km
        :type box_size: int, float, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:
        Plot the array configuration from the antenna positions.
        """

        pathlib.Path(destination).mkdir(exist_ok=True)
        input_params = locals()
        plot_array_configuration(input_params, self.root, "point")
