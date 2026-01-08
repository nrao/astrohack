import xarray as xr
import pathlib

from typing import List, Union

import toolviper.utils.logger as logger

from toolviper.utils.parameter import validate

from .base_mds import AstrohackBaseFile

from astrohack.core.beamcut import (
    plot_beamcut_in_amplitude_chunk,
    plot_beamcut_in_attenuation_chunk,
    create_report_chunk,
    plot_cuts_in_lm_chunk,
)
from astrohack.utils import get_method_list_string
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


class AstrohackBeamcutFile(AstrohackBaseFile):
    """Data class for beam cut data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackBeamcutFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBeamcutFile object
        :rtype: AstrohackBeamcutFile
        """
        super().__init__(file=file)

    @validate(custom_checker=custom_unit_checker)
    def observation_summary(
        self,
        summary_file: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[str, int, List[int]] = "all",
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
        ddi: Union[str, int, List[int]] = "all",
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
        ddi: Union[str, int, List[int]] = "all",
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
        ddi: Union[str, int, List[int]] = "all",
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
        ddi: Union[str, int, List[int]] = "all",
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
