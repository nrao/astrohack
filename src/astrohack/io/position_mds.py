import pathlib

from typing import List, Union

import toolviper.utils.logger as logger

from astrohack.antenna import get_proper_telescope
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.core.locit_2 import (
    export_position_xds_to_table_row,
    export_position_xds_to_parminator,
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
