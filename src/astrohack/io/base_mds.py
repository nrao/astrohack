import xarray as xr

import toolviper.utils.logger as logger

from astrohack.utils import (
    add_caller_and_version_to_dict_2,
    get_summary_header,
    get_property_string,
    get_data_content_string,
)
from astrohack.utils.text import (
    print_summary_header,
    print_dict_table,
    get_method_list_string,
    print_data_contents,
)


class AstrohackBaseFile:
    """Base Data class for astrohack.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackBeamcutFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBeamcutFile object
        :rtype: AstrohackBeamcutFile
        """
        self.file = file
        self._file_is_open = False
        self.root = None

    def __getitem__(self, key: str) -> xr.DataTree:
        """
        get item implementation that gets the xdtree at key.

        :param key: Key for which to fetch a subtree
        :type key: str

        :return: corresponding subtree
        :rtype: xr.DataTree
        """
        return self.root[key]

    def __setitem__(self, key: str, subtree: xr.DataTree) -> None:
        """
        Set item implementation that sets the xdtree at key.

        :param key: Key for which to set a subtree
        :type key: str

        :param subtree: Subtree to attach at key
        :type subtree: xr.DataTree

        :return: None
        :rtype: NoneType
        """
        self.root[key] = subtree
        return

    @property
    def is_open(self) -> bool:
        """
        Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def keys(self, *args, **kwargs):
        """
        Get children keys

        :param args: args to deliver to dict.keys() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.keys() method
        :type kwargs: dict

        :return: dict keys iterable
        :rtype: dict_keys
        """
        return self.root.children.keys(*args, **kwargs)

    def open(self, file: str = None) -> bool:
        """
        Open beamcut file.

        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            # Chunks='auto' means lazy dask loading with automatic choice of chunk size
            # chunks=None is direct opening.
            self.root = xr.open_datatree(file, engine="zarr", chunks="auto")

            self._file_is_open = True
            self.file = file

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def write(self):
        """
        Write mds to disk by saving the data tree to a file
        """
        self.root.to_zarr(self.file, mode="w", consolidated=True)

    def summary(self) -> None:
        """
        Prints summary of the AstrohackBeamcutFile object, with available data, attributes and available methods

        :return: None
        :rtype: NoneType
        """
        outstr = get_summary_header(self.file)
        outstr += get_property_string(self.root.attrs)
        outstr += get_method_list_string(self)
        outstr += get_data_content_string(self.root)
        print(outstr)

    @classmethod
    def create_from_input_parameters(cls, file_name: str, input_parameters: dict):
        data_obj = cls(file_name)
        data_obj.root = xr.DataTree(name="root")
        add_caller_and_version_to_dict_2(data_obj.root.attrs, direct_call=False)
        data_obj.root.attrs["input_parameters"] = input_parameters
        return data_obj

    #
    # def observation_summary(
    #     self,
    #     summary_file: str,
    #     ant: Union[str, List[str]] = "all",
    #     ddi: Union[str, int, List[int]] = "all",
    #     az_el_key: str = "center",
    #     phase_center_unit: str = "radec",
    #     az_el_unit: str = "deg",
    #     time_format: str = "%d %h %Y, %H:%M:%S",
    #     tab_size: int = 3,
    #     print_summary: bool = True,
    #     parallel: bool = False,
    # ) -> None:
    #     """
    #     Create a Summary of observation information
    #
    #     :param summary_file: Text file to put the observation summary
    #     :type summary_file: str
    #
    #     :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
    #     :type ant: list or str, optional
    #
    #     :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
    #     :type ddi: list or int, optional
    #
    #     :param az_el_key: What type of Azimuth & Elevation information to print, 'mean', 'median' or 'center', default\
    #     is 'center'
    #     :type az_el_key: str, optional
    #
    #     :param phase_center_unit: What unit to display phase center coordinates, 'radec' and angle units supported, \
    #     default is 'radec'
    #     :type phase_center_unit: str, optional
    #
    #     :param az_el_unit: Angle unit used to display Azimuth & Elevation information, default is 'deg'
    #     :type az_el_unit: str, optional
    #
    #     :param time_format: datetime time format for the start and end dates of observation, default is \
    #     "%d %h %Y, %H:%M:%S"
    #     :type time_format: str, optional
    #
    #     :param tab_size: Number of spaces in the tab levels, default is 3
    #     :type tab_size: int, optional
    #
    #     :param print_summary: Print the summary at the end of execution, default is True
    #     :type print_summary: bool, optional
    #
    #     :param parallel: Run in parallel, defaults to False
    #     :type parallel: bool, optional
    #
    #     :return: None
    #     :rtype: NoneType
    #
    #     **Additional Information**
    #
    #     This method produces a summary of the data in the AstrohackBeamcutFile displaying general information,
    #     spectral information, beam image characteristics and aperture image characteristics.
    #     """
