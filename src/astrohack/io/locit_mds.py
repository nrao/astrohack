from astrohack import get_proper_telescope
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils import (
    create_pretty_table,
    rad_to_hour_str,
    rad_to_deg_str,
    compute_antenna_relative_off,
    notavail,
)
from astrohack.utils.tools import get_telescope_lat_lon_rad


class AstrohackLocitFile2(AstrohackBaseFile):
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

    # @toolviper.utils.parameter.validate()
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
