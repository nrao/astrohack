import numpy as np
import pathlib

from typing import Union, Tuple, List

import toolviper.utils.parameter

from astrohack.antenna import get_proper_telescope
from astrohack.core.extract_locit import (
    plot_source_table,
)
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils import (
    create_pretty_table,
    rad_to_hour_str,
    rad_to_deg_str,
    compute_antenna_relative_off,
    notavail,
)
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.utils.validation import custom_unit_checker
from astrohack.visualization.diagnostics import plot_array_configuration


class AstrohackLocitFile(AstrohackBaseFile):
    """Data class for locit data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackLocitFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackLocitFile object
        :rtype: AstrohackLocitFile
        """
        super().__init__(file=file)

    def print_source_table(self) -> None:
        """Prints a table with the sources observed for antenna location determination"""
        print("\nSources:")
        field_names = [
            "Id",
            "Name",
            "RA FK5",
            "DEC FK5",
            "RA precessed",
            "DEC precessed",
        ]
        table = create_pretty_table(field_names, "l")
        src_dict = self.root.attrs["source_dict"]

        for source in src_dict.values():
            table.add_row(
                [
                    source["id"],
                    source["name"],
                    rad_to_hour_str(source["fk5"][0]),
                    rad_to_deg_str(source["fk5"][1]),
                    rad_to_hour_str(source["precessed"][0]),
                    rad_to_deg_str(source["precessed"][1]),
                ]
            )
        print(table)

    @toolviper.utils.parameter.validate()
    def print_array_configuration(self, relative: bool = True) -> None:
        """Prints a table containing the array configuration

        :param relative: Print antenna coordinates relative to array center or in geocentric coordinates, default is True
        :type relative: bool, optional

        .. _Description:

        Print arrayx configuration in the dataset. Also marks the reference antenna and the antennas that are
        absent from the dataset. Coordinates of antenna stations can be relative to the array center or Geocentric
        (longitude, latitude and radius)

        """

        telescope_name = self.root.attrs["telescope_name"]
        telescope = get_proper_telescope(telescope_name)

        print(f"\n{telescope_name} antennas, # of antennas {len(self.root.keys())}:")
        if relative:
            nfields = 5
            field_names = [
                "Name",
                "Station",
                "East [m]",
                "North [m]",
                "Elevation [m]",
                "Distance [m]",
            ]
            tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
        else:
            nfields = 4
            field_names = ["Name", "Station", "Longitude", "Latitude", "Radius [m]"]
            tel_lon, tel_lat, tel_rad = None, None, None

        table = create_pretty_table(field_names)
        for ant_name in telescope.antenna_list:
            row = [ant_name]
            try:
                ant_key = f"ant_{ant_name}"
                ant_info = self.root[ant_key].attrs["antenna_info"]
                row.append(ant_info["station"])
                if relative:
                    offsets = compute_antenna_relative_off(
                        ant_info, tel_lon, tel_lat, tel_rad
                    )
                    row.extend(
                        [
                            f"{offsets[0]:.4f}",
                            f"{offsets[1]:.4f}",
                            f"{offsets[2]:.4f}",
                            f"{offsets[3]:.4f}",
                        ]
                    )
                else:
                    row.extend(
                        [
                            rad_to_deg_str(ant_info["longitude"]),
                            rad_to_deg_str(ant_info["latitude"]),
                            f'{ant_info["radius"]:.4f}',
                        ]
                    )
            except KeyError:
                for i_col in range(nfields):
                    row.append(notavail)

            table.add_row(row)

        print(table)
        return

    @toolviper.utils.parameter.validate()
    def plot_source_positions(
        self,
        destination: str,
        labels: bool = True,
        precessed: bool = False,
        display: bool = False,
        figure_size: Union[Tuple, List[float], np.array] = None,
        dpi: int = 300,
    ) -> None:
        """Plot source positions in either FK5 or precessed right ascension and declination.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param labels: Add source labels to the plot, defaults to False
        :type labels: bool, optional

        :param precessed: Plot in precessed coordinates? defaults to False (FK5)
        :type precessed: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:

        Plot the sources on the source list to a full 24 hours 180 degrees flat 2D representation of the full sky.
        If precessed is set to True the coordinates precessd to the midpoint of the observations is plotted, otherwise
        the FK5 coordinates are plotted.
        The source names can be plotted next to their positions if label is True, however plots may become too crowded
        if that is the case.

        """
        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        if precessed:
            filename = str(
                pathlib.Path(destination).joinpath("locit_source_table_precessed.png")
            )
            time_range = self.root.attrs["time_range"]
            obs_midpoint = (time_range[1] + time_range[0]) / 2.0

        else:
            filename = str(
                pathlib.Path(destination).joinpath("locit_source_table_fk5.png")
            )
            obs_midpoint = None

        plot_source_table(
            filename,
            self.root.attrs["source_dict"],
            precessed=precessed,
            obs_midpoint=obs_midpoint,
            display=display,
            figure_size=figure_size,
            dpi=dpi,
            label=labels,
        )

        return

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_array_configuration(
        self,
        destination: str,
        stations: bool = True,
        zoff: bool = False,
        unit: str = "m",
        box_size: Union[int, float] = 5000,
        display: bool = False,
        figure_size: Union[Tuple, List[float], np.array] = None,
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


        """
        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        plot_array_configuration(param_dict, self.root, "locit")
        return
