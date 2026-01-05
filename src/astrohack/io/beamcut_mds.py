import xarray as xr
import pathlib

from typing import List, Union

import toolviper.utils.logger as logger

from toolviper.utils.parameter import validate

from astrohack.core.beamcut import (
    plot_beamcut_in_amplitude_chunk,
    plot_beamcut_in_attenuation_chunk,
    create_report_chunk,
    plot_cuts_in_lm_chunk,
)
from astrohack.utils import print_method_list_xdt
from astrohack.utils.text import (
    print_summary_header,
    print_dict_table,
    print_method_list,
    print_data_contents,
)
from astrohack.visualization.textual_data import (
    generate_observation_summary_for_beamcut,
)
from astrohack.utils.graph import compute_graph
from astrohack.utils.validation import custom_plots_checker, custom_unit_checker


class AstrohackBeamcutFile:

    def __init__(self, file: str):
        """Initialize an AstrohackBeamcutFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBeamcutFile object
        :rtype: AstrohackBeamcutFile
        """
        self.file = file
        self._file_is_open = False
        self._input_pars = None
        self.xdt = None

    def __getitem__(self, key: str) -> xr.DataTree:
        """
        get item implementation that gets the xdtree at key.

        :param key: Key for which to fetch a subtree
        :type key: str

        :return: corresponding subtree
        :rtype: xr.DataTree
        """
        return self.xdt[key]

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
        self.xdt[key] = subtree
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
        return self.xdt.children.keys(*args, **kwargs)

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
            self.xdt = xr.open_datatree(file, engine="zarr", chunks="auto")
            self._input_pars = self.xdt.attrs

            self._file_is_open = True
            self.file = file

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def summary(self) -> None:
        """
        Prints summary of the AstrohackBeamcutFile object, with available data, attributes and available methods

        :return: None
        :rtype: NoneType
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["Antenna", "DDI", "Cut"])
        print_method_list_xdt(self)

    @validate(custom_checker=custom_unit_checker)
    def observation_summary(
        self,
        summary_file: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        az_el_key: str = "center",
        phase_center_unit: str = "radec",
        az_el_unit: str = "deg",
        time_format: str = "%d %h %Y, %H:%M:%S",
        tab_size: int = 3,
        print_summary: bool = True,
        parallel: bool = False,
    ) -> None:
        """
        Create a Summary of observation information

        :param summary_file: Text file to put the observation summary
        :type summary_file: str

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param az_el_key: What type of Azimuth & Elevation information to print, 'mean', 'median' or 'center', default\
        is 'center'
        :type az_el_key: str, optional

        :param phase_center_unit: What unit to display phase center coordinates, 'radec' and angle units supported, \
        default is 'radec'
        :type phase_center_unit: str, optional

        :param az_el_unit: Angle unit used to display Azimuth & Elevation information, default is 'deg'
        :type az_el_unit: str, optional

        :param time_format: datetime time format for the start and end dates of observation, default is \
        "%d %h %Y, %H:%M:%S"
        :type time_format: str, optional

        :param tab_size: Number of spaces in the tab levels, default is 3
        :type tab_size: int, optional

        :param print_summary: Print the summary at the end of execution, default is True
        :type print_summary: bool, optional

        :param parallel: Run in parallel, defaults to False
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType

        **Additional Information**

        This method produces a summary of the data in the AstrohackBeamcutFile displaying general information,
        spectral information, beam image characteristics and aperture image characteristics.
        """

        param_dict = locals()
        key_order = ["ant", "ddi"]
        execution, summary_list = compute_graph(
            self,
            generate_observation_summary_for_beamcut,
            param_dict,
            key_order,
            parallel,
            fetch_returns=True,
        )
        full_summary = "".join(summary_list)
        with open(summary_file, "w") as output_file:
            output_file.write(full_summary)
        if print_summary:
            print(full_summary)

    @validate(custom_checker=custom_plots_checker)
    def plot_beamcut_in_amplitude(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        y_scale: list[float] = None,
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds in amplitude

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param y_scale: Set the y scale for the plots.
        :type y_scale: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        compute_graph(
            self,
            plot_beamcut_in_amplitude_chunk,
            param_dict,
            ["ant", "ddi"],
            parallel=parallel,
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def plot_beamcut_in_attenuation(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        y_scale: str = None,
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds in attenuation

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param y_scale: Set the y scale for the plots.
        :type y_scale: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        compute_graph(
            self,
            plot_beamcut_in_attenuation_chunk,
            param_dict,
            ["ant", "ddi"],
            parallel=parallel,
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def plot_beam_cuts_over_sky(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds over the sky

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        compute_graph(
            self,
            plot_cuts_in_lm_chunk,
            param_dict,
            ["ant", "ddi"],
            parallel=parallel,
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def create_beam_fit_report(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        parallel: bool = False,
    ) -> None:
        """
        Create reports on the parameters of the gaussians fitted to the beamcut.

        :param destination: Directory into which to save the reports.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param parallel: run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        compute_graph(
            self, create_report_chunk, param_dict, ["ant", "ddi"], parallel=parallel
        )
        return
