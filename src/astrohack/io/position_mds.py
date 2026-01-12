import pathlib
import numpy as np

from typing import List, Union, Tuple

import toolviper.utils.logger as logger

from astrohack.antenna import get_proper_telescope
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.core.locit import (
    export_position_xds_to_table_row,
    export_position_xds_to_parminator,
    plot_sky_coverage_chunk,
    plot_delays_chunk,
    plot_antenna_position_corrections_worker,
)
from astrohack.utils import (
    convert_unit,
    clight,
    notavail,
    create_pretty_table,
    param_to_list,
    add_prefix,
    string_to_ascii_file,
)
from astrohack.utils.graph import compute_graph


class AstrohackPositionFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackPositionFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPositionFile object
        :rtype: AstrohackPositionFile
        """
        super().__init__(file=file)

    # @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def export_locit_fit_results(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[str, int, List[int]] = "all",
        position_unit: str = "m",
        time_unit: str = "hour",
        delay_unit: str = "nsec",
        phase_unit: str = "deg",
    ) -> None:
        """Export antenna position fit results to a text file.

        :param destination: Name of the destination folder to contain exported fit results
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param position_unit: Unit to list position fit results, defaults to 'm'
        :type position_unit: str, optional

        :param time_unit: Unit for time in position fit results, defaults to 'hour'
        :type time_unit: str, optional

        :param delay_unit: Unit for delays, defaults to 'nsec'
        :type delay_unit: str, optional

        :param phase_unit: Unit for phasess, defaults to 'deg'
        :type phase_unit: str, optional

        .. _Description:

        Produce a text file with the fit results from astrohack.locit for better determination of antenna locations.
        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        len_fact = convert_unit("m", position_unit, "length")
        del_fact = convert_unit("sec", delay_unit, kind="time")
        pha_fact = convert_unit("rad", phase_unit, kind="trigonometric")
        pos_fact = len_fact * clight
        combined = self.root.attrs["combined"]
        input_pars = self.root.attrs["input_parameters"]

        if combined:
            field_names = [
                "Antenna",
                "Station",
                f"RMS [{delay_unit}]",
                f"RMS [{phase_unit}]",
                f"F. delay [{delay_unit}]",
                f"X offset [{position_unit}]",
                f"Y offset [{position_unit}]",
                f"Z offset [{position_unit}]",
            ]
            specifier = f"combined_{input_pars['combine_ddis']}"

        else:
            field_names = [
                "Antenna",
                "Station",
                "DDI",
                f"RMS [{delay_unit}]",
                f"RMS [{phase_unit}]",
                f"F. delay [{delay_unit}]",
                f"X offset [{position_unit}]",
                f"Y offset [{position_unit}]",
                f"Z offset [{position_unit}]",
            ]
            specifier = "separated_ddis"
        kterm_present = input_pars["fit_kterm"]
        rate_present = input_pars["fit_delay_rate"]
        if kterm_present:
            field_names.extend([f"K offset [{position_unit}]"])
        if rate_present:
            slo_unit = f"{delay_unit}/{time_unit}"
            slo_fact = del_fact / convert_unit("day", time_unit, "time")
            field_names.extend([f"Rate [{slo_unit}]"])
        else:
            slo_unit = notavail
            slo_fact = 1.0

        table = create_pretty_table(field_names)
        telescope = get_proper_telescope(self.root.attrs["telescope_name"])
        full_antenna_list = telescope.antenna_list
        selected_antenna_list = param_to_list(ant, self, "ant")

        for ant_name in full_antenna_list:
            ant_key = add_prefix(ant_name, "ant")
            if ant_name == self.root.attrs["reference_antenna"]:
                ant_name += " (ref)"

            if ant_key in selected_antenna_list:
                if ant_key in self.keys():
                    antenna = self[ant_key]
                    if combined:
                        row = [ant_name, antenna.attrs["antenna_info"]["station"]]
                        table.add_row(
                            export_position_xds_to_table_row(
                                row,
                                antenna.attrs,
                                del_fact,
                                pha_fact,
                                pos_fact,
                                slo_fact,
                                position_unit,
                                delay_unit,
                                kterm_present,
                                rate_present,
                            )
                        )
                    else:
                        ddi_list = param_to_list(ddi, self[ant_key], "ddi")
                        for ddi_key in ddi_list:
                            row = [
                                ant_name,
                                antenna[ddi_key].attrs["antenna_info"]["station"],
                                ddi_key.split("_")[1],
                            ]
                            table.add_row(
                                export_position_xds_to_table_row(
                                    row,
                                    antenna[ddi_key].attrs,
                                    del_fact,
                                    pha_fact,
                                    pos_fact,
                                    slo_fact,
                                    position_unit,
                                    delay_unit,
                                    kterm_present,
                                    rate_present,
                                )
                            )

        print(table.get_string())
        string_to_ascii_file(
            table.get_string(),
            f"{destination}/position_{specifier}_fit_results.txt",
        )

    # @toolviper.utils.parameter.validate()
    def export_results_to_parminator(
        self,
        filename: str,
        ant: Union[str, List[str]] = "all",
        ddi: int = None,
        correction_threshold: float = 0.01,
    ) -> None:
        """Export antenna position fit results to a VLA parminator file.

        :param filename: Name of the parminator file to be created
        :type filename: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param correction_threshold: Correction threshold in meters to include an antenna position correction in output.
        :type correction_threshold: float, optional

        .. _Description:

        Produce a VLA parminator compatible text file with the fit results from astrohack.locit.
        """
        param_dict = locals()
        combined = self.root.attrs["combined"]
        input_pars = self.root.attrs["input_parameters"]

        if (not combined) and (not isinstance(ddi, int)):
            msg = "If position file contains multiple DDIs one must be specified."
            logger.error(msg)
            raise ValueError(msg)

        kterm_present = input_pars["fit_kterm"]

        telescope = get_proper_telescope(self.root.attrs["telescope_name"])
        full_antenna_list = telescope.antenna_list
        selected_antenna_list = param_to_list(ant, self, "ant")
        threshold = correction_threshold

        parmstr = ""
        for ant_name in full_antenna_list:
            ant_key = add_prefix(ant_name, "ant")

            if ant_key in selected_antenna_list:
                if ant_key in self.keys():
                    if combined:
                        position_xds = self[ant_key]
                    else:
                        position_xds = self[ant_key][f"ddi_{ddi}"]

                    parmstr += export_position_xds_to_parminator(
                        position_xds.attrs, threshold, kterm_present
                    )

        string_to_ascii_file(parmstr, filename)

    # @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_sky_coverage(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[str, int, List[int]] = "all",
        time_unit: str = "hour",
        angle_unit: str = "deg",
        display: bool = False,
        figure_size: Union[Tuple, List[float], np.array] = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """Plot the sky coverage of the data used for antenna position fitting

        :param destination: Name of the destination folder to contain the plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for angle in plots, defaults to 'deg'
        :type angle_unit: str, optional

        :param time_unit: Unit for time in plots, defaults to 'hour'
        :type time_unit: str, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot size in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: plot resolution in pixels per inch, default is 300
        :type dpi: int, optional

        :param parallel: If True will use an existing astrohack client to produce plots in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        This method produces 4 plots for each selected antenna and DDI. These plots are:
        1) Time vs Elevation
        2) Time vs Hour Angle
        3) Time vs Declination
        4) Hour Angle vs Declination

        These plots are intended to display the coverage of the sky of the fitted data

        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        param_dict["combined"] = self.root.attrs["combined"]

        if self.root.attrs["combined"]:
            compute_graph(
                self, plot_sky_coverage_chunk, param_dict, ["ant"], parallel=parallel
            )
        else:
            compute_graph(
                self,
                plot_sky_coverage_chunk,
                param_dict,
                ["ant", "ddi"],
                parallel=parallel,
            )

    # @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_delays(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[str, int, List[int]] = "all",
        time_unit: str = "hour",
        angle_unit: str = "deg",
        delay_unit: str = "nsec",
        plot_model: bool = True,
        display: bool = False,
        figure_size: Union[Tuple, List[float], np.array] = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """Plot the delays used for antenna position fitting and optionally the resulting fit.

        :param destination: Name of the destination folder to contain the plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for angle in plots, defaults to 'deg'
        :type angle_unit: str, optional

        :param time_unit: Unit for time in plots, defaults to 'hour'
        :type time_unit: str, optional

        :param delay_unit: Unit for delay in plots, defaults to 'nsec'
        :type delay_unit: str, optional

        :param plot_model: Plot the fitted model results alongside the data.
        :type plot_model: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot size in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: plot resolution in pixels per inch, default is 300
        :type dpi: int, optional

        :param parallel: If True will use an existing astrohack client to produce plots in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        This method produces 4 plots for each selected antenna and DDI. These plots are:
        1) Time vs Delays
        2) Elevation vs Delays
        3) Hour Angle vs Delays
        4) Declination vs Delays

        These plots are intended to display the gain variation with the 4 relevant parameters for the fitting and also
        asses the quality of the position fit.

        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        param_dict["combined"] = self.root.attrs["combined"]
        param_dict["comb_type"] = self.root.attrs["input_parameters"]["combine_ddis"]
        if self.root.attrs["combined"]:
            compute_graph(
                self, plot_delays_chunk, param_dict, ["ant"], parallel=parallel
            )
        else:
            compute_graph(
                self, plot_delays_chunk, param_dict, ["ant", "ddi"], parallel=parallel
            )

    # @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_position_corrections(
        self,
        destination: str,
        ant: Union[str, List[str]] = "all",
        ddi: Union[str, int, List[int]] = "all",
        unit: str = "km",
        box_size: Union[int, float] = 5,
        scaling: Union[int, float] = 250,
        figure_size: Union[Tuple, List[float], np.array] = None,
        display: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot Antenna position corrections on an array configuration plot

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param ant: Select which antennas are to be plotted, defaults to all when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param unit: Unit for the plot, valid values are length units, default is km
        :type unit: str, optional

        :param box_size: Size of the box for plotting the inner part of the array in unit, default is 5 km
        :type box_size: int, float, optional

        :param scaling: scaling factor to plotting the corrections, default is 250
        :type scaling: int, float, optional

        :param display: Display plots inline or suppress, defaults to False
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:

        Plot the position corrections computed by locit on top of an array configuration plot.
        The corrections are too small to be visualized on the array plot since they are of the order of mm and the array
        is usually spread over km, or at least hundreds of meters.
        The scaling factor is used to bring the corrections to a scale discernible on the plot, this plot should not be
        used to estimate correction values, for that purpose use export_locit_fit_results instead.

        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        combined = self.root.attrs["combined"]
        telescope = get_proper_telescope(
            self.root.attrs["telescope_name"], param_dict["ant"]
        )
        ref_ant = self.root.attrs["reference_antenna"]

        ant_list = param_to_list(ant, self, "ant")
        if combined:
            filename = (
                f"{destination}/position_corrections_combined_"
                + f'{self.root.attrs["input_parameters"]["combine_ddis"]}.png'
            )
            attribute_list = []
            for ant in ant_list:
                attribute_list.append(self[ant].attrs)
            plot_antenna_position_corrections_worker(
                attribute_list, filename, telescope, ref_ant, param_dict
            )

        else:
            ddi_list = []
            if ddi == "all":
                for ant in ant_list:
                    ddi_list.extend(self[ant].keys())
                ddi_list = np.unique(ddi_list)
            else:
                ddi_list = ddi
                for i_ddi in range(len(ddi_list)):
                    ddi_list[i_ddi] = "ddi_" + ddi_list[i_ddi]
            for ddi in ddi_list:
                filename = f"{destination}/position_corrections_separated_{ddi}.png"
                attribute_list = []
                for ant in ant_list:
                    if ddi in self[ant].keys():
                        attribute_list.append(self[ant][ddi].attrs)
                plot_antenna_position_corrections_worker(
                    attribute_list, filename, telescope, ref_ant, param_dict
                )
