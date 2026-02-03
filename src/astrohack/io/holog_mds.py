from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackHologFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackHologFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackHologFile object
        :rtype: AstrohackHologFile
        """
        super().__init__(file=file)

    # @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    # def plot_diagnostics(
    #     self,
    #     destination: str,
    #     delta: float = 0.01,
    #     ant: Union[str, List[str]] = "all",
    #     ddi: Union[str, int, List[int]] = "all",
    #     map_id: Union[int, List[int]] = "all",
    #     complex_split: str = "polar",
    #     display: bool = False,
    #     figure_size: Union[Tuple, List[float], np.array] = None,
    #     dpi: int = 300,
    #     parallel: bool = False,
    # ) -> None:
    #     """ Plot diagnostic calibration plots from the holography data file.
    #
    #     :param destination: Name of the destination folder to contain diagnostic plots
    #     :type destination: str
    #     :param delta: Defines a fraction of cell_size around which to look for peaks., defaults to 0.01
    #     :type delta: float, optional
    #     :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
    #     :type ant: list or str, optional
    #     :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
    #     :type ddi: list or int, optional
    #     :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
    #     configuration,  defaults to "all" when None, ex. 0
    #     :type map_id: list or int, optional
    #     :param complex_split: How to split complex data, cartesian (real + imaginary) or polar (amplitude + phase), \
    #     default is polar
    #     :type complex_split: str, optional
    #     :param display: Display plots inline or suppress, defaults to True
    #     :type display: bool, optional
    #     :param figure_size: 2 element array/list/tuple with the plot sizes in inches
    #     :type figure_size: numpy.ndarray, list, tuple, optional
    #     :param dpi: dots per inch to be used in plots, default is 300
    #     :type dpi: int, optional
    #     :param parallel: Run in parallel, defaults to False
    #     :type parallel: bool, optional
    #
    #     **Additional Information**
    #     The visibilities extracted by extract_holog are complex due to the nature of interferometric measurements. To
    #     ease the visualization of the complex data it can be split into real and imaginary parts (cartesian) or in
    #     amplitude and phase (polar).
    #
    #     .. rubric:: Available complex splitting possibilities:
    #     - *cartesian*: Split is done to a real part and an imaginary part in the plots
    #     - *polar*:     Split is done to an amplitude and a phase in the plots
    #
    #     """
    #
    #     param_dict = locals()
    #     param_dict["map"] = map_id
    #
    #     pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
    #     key_order = ["ddi", "map", "ant"]
    #     compute_graph(self, calibration_plot_chunk, param_dict, key_order, parallel)
    #
    # @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    # def plot_lm_sky_coverage(
    #     self,
    #     destination: str,
    #     ant: Union[str, List[str]] = "all",
    #     ddi: Union[str, int, List[int]] = "all",
    #     map_id: Union[int, List[int]] = "all",
    #     angle_unit: str = "deg",
    #     time_unit: str = "hour",
    #     plot_correlation: Union[str, List[str]] = None,
    #     complex_split: str = "polar",
    #     phase_unit: str = "deg",
    #     display: bool = False,
    #     figure_size: Union[Tuple, List[float], np.array] = None,
    #     dpi: int = 300,
    #     parallel: bool = False,
    # ) -> None:
    #     """ Plot directional cosine coverage.
    #
    #     :param destination: Name of the destination folder to contain plots
    #     :type destination: str
    #     :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
    #     :type ant: list or str, optional
    #     :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
    #     :type ddi: list or int, optional
    #     :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
    #     configuration,  defaults to "all" when None, ex. 0
    #     :type map_id: list or int, optional
    #     :param angle_unit: Unit for L and M axes in plots, default is 'deg'.
    #     :type angle_unit: str, optional
    #     :param time_unit: Unit for time axis in plots, default is 'hour'.
    #     :type time_unit: str, optional
    #     :param plot_correlation: Which correlation to plot against L and M, default is None (no correlation plots).
    #     :type plot_correlation: str, list, optional
    #     :param complex_split: How to split complex data, cartesian (real + imaginary) or polar (amplitude + phase), \
    #     default is polar
    #     :type complex_split: str, optional
    #     :param phase_unit: Unit for phase in 'polar' plots, default is 'deg'.
    #     :type phase_unit: str
    #     :param display: Display plots inline or suppress, defaults to True
    #     :type display: bool, optional
    #     :param figure_size: 2 element array/list/tuple with the plot sizes in inches
    #     :type figure_size: numpy.ndarray, list, tuple, optional
    #     :param dpi: dots per inch to be used in plots, default is 300
    #     :type dpi: int, optional
    #     :param parallel: Run in parallel, defaults to False
    #     :type parallel: bool, optional
    #
    #     **Additional Information**
    #     The visibilities extracted by extract_holog are complex due to the nature of interferometric measurements. To
    #     ease the visualization of the complex data it can be split into real and imaginary parts (cartesian) or in
    #     amplitude and phase (polar).
    #
    #     .. rubric:: Available complex splitting possibilities:
    #     - *cartesian*: Split is done to a real part and an imaginary part in the plots
    #     - *polar*:     Split is done to an amplitude and a phase in the plots
    #
    #     .. rubric:: Plotting correlations:
    #     - *RR, RL, LR, LL*: Are available for circular systems
    #     - *XX, XY, YX, YY*: Are available for linear systems
    #     - *all*: Plot all correlations in dataset
    #
    #     """
    #
    #     param_dict = locals()
    #     param_dict["map"] = map_id
    #
    #     pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
    #     key_order = ["ddi", "map", "ant"]
    #     compute_graph(self, plot_lm_coverage, param_dict, key_order, parallel)
    #     return
    #
    # @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    # def export_to_aips(
    #     self,
    #     destination: str,
    #     ant: Union[str, List[str]] = "all",
    #     ddi: Union[str, int, List[int]] = "all",
    #     map_id: Union[int, List[int]] = "all",
    #     parallel: bool = False,
    # ) -> None:
    #     """ Export data compatible to AIPS's HOLOG task
    #
    #     :param destination: Name of the destination folder to contain SCII files
    #     :type destination: str
    #     :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
    #     :type ant: list or str, optional
    #     :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
    #     :type ddi: list or int, optional
    #     :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
    #     configuration,  defaults to "all" when None, ex. 0
    #     :type map_id: list or int, optional
    #     :param parallel: Run in parallel, defaults to False
    #     :type parallel: bool, optional
    #
    #     **Additional Information**
    #
    #     This method converts the data for an Antenna mapping to the ASCII format used by AIPS's HOLOG task.
    #     Currently only stokes I is supported.
    #     """
    #     param_dict = locals()
    #     param_dict["map"] = map_id
    #
    #     pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
    #     key_order = ["ddi", "map", "ant"]
    #     compute_graph(self, export_to_aips, param_dict, key_order, parallel)
    #     return

    # @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    # def observation_summary(
    #     self,
    #     summary_file: str,
    #     ant: Union[str, List[str]] = "all",
    #     ddi: Union[str, int, List[int]] = "all",
    #     map_id: Union[int, List[int]] = "all",
    #     az_el_key: str = "center",
    #     phase_center_unit: str = "radec",
    #     az_el_unit: str = "deg",
    #     time_format: str = "%d %h %Y, %H:%M:%S",
    #     tab_size: int = 3,
    #     print_summary: bool = True,
    #     parallel: bool = False,
    # ) -> None:
    #     """ Create a Summary of observation information
    #
    #     :param summary_file: Text file to put the observation summary
    #     :type summary_file: str
    #     :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
    #     :type ant: list or str, optional
    #     :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
    #     :type ddi: list or int, optional
    #     :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
    #     configuration,  defaults to "all" when None, ex. 0
    #     :type map_id: list or int, optional
    #     :param az_el_key: What type of Azimuth & Elevation information to print, 'mean', 'median' or 'center', default\
    #     is 'center'
    #     :type az_el_key: str, optional
    #     :param phase_center_unit: What unit to display phase center coordinates, 'radec' and angle units supported, \
    #     default is 'radec'
    #     :type phase_center_unit: str, optional
    #     :param az_el_unit: Angle unit used to display Azimuth & Elevation information, default is 'deg'
    #     :type az_el_unit: str, optional
    #     :param time_format: datetime time format for the start and end dates of observation, default is \
    #     "%d %h %Y, %H:%M:%S"
    #     :type time_format: str, optional
    #     :param tab_size: Number of spaces in the tab levels, default is 3
    #     :type tab_size: int, optional
    #     :param print_summary: Print the summary at the end of execution, default is True
    #     :type print_summary: bool, optional
    #     :param parallel: Run in parallel, defaults to False
    #     :type parallel: bool, optional
    #
    #     **Additional Information**
    #
    #     This method produces a summary of the data in the AstrohackHologFile displaying general information,
    #     spectral information and suggested beam image characteristics.
    #     """
    #
    #     param_dict = locals()
    #     param_dict["map"] = map_id
    #     key_order = ["ddi", "map", "ant"]
    #     execution, summary = compute_graph(
    #         self,
    #         generate_observation_summary,
    #         param_dict,
    #         key_order,
    #         parallel,
    #         fetch_returns=True,
    #     )
    #     summary = "".join(summary)
    #     with open(summary_file, "w") as output_file:
    #         output_file.write(summary)
    #     if print_summary:
    #         print(summary)
