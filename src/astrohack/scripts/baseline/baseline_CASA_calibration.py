import argparse
import time
import numpy as np

import casatools

from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    file_is_asdm,
    run_casatask,
    proceed_check,
    list_input_tooltip,
)
from astrohack.utils.text import format_duration


def parse():
    desc = "CASA pre-locit script\n"
    desc += "Execute fringe fit, averaging and phase cal to produce the cal table to ingested by astrohack's locit"

    parser = argparse.ArgumentParser(
        description=f"{desc}", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("filename", type=str, help="Path to the input MS/ASDM file")

    parser.add_argument("refant", type=str, help="Reference antenna for calibration")

    parser.add_argument(
        "-r",
        "--root-name",
        type=str,
        default=None,
        help="Root name for the calibration tables, default is filename without extension",
    )

    parser.add_argument(
        "-f",
        "--fringefit_source",
        default="0319+415",
        help="Fringe fit source, default is 0319+415",
    )
    parser.add_argument(
        "-S",
        "--scans_to_flag",
        default=None,
        type=str,
        help="Comma separated list of scans to flag, default is None",
    )

    parser.add_argument(
        "-i",
        "--intent",
        default="CALIBRATE_POINTING#ON_SOURCE",
        type=str,
        help="Intent for pointing observations.",
    )

    parser.add_argument(
        "-s",
        "--spectral-window",
        type=str,
        default="all",
        help=f"Select SPWs for locit processing, {list_input_tooltip('0,1,2')}, default is %(default)s",
    )

    parser.add_argument(
        "-a",
        "--antenna",
        default="all",
        help="Select antennas for which to produce antenna position corrections, "
        f"{list_input_tooltip('ea01,ea02')}, default is %(default)s"
        "",
    )

    parser.add_argument(
        "-e",
        "--elevation-limit",
        type=float,
        default=10.0,
        help="Lowest elevation of data for consideration in degrees, default is %(default).1f",
    )

    parser.add_argument(
        "-p",
        "--polarization",
        type=str,
        choices=["both", "L", "R"],
        default="both",
        help="Which polarization hands to be used for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-c",
        "--combination",
        type=str,
        choices=["simple, difference, no"],
        default="simple",
        help="How to combine different spws for locit processing, default is %(default)s",
    )

    parser.add_argument(
        "-k",
        "--fit_kterm",
        action="store_true",
        default=False,
        help="Fit antennas K term (i.e. Offset between azimuth and elevation axes)",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing files (MSes, caltables, locit files, plots)",
    )

    parser.add_argument(
        "--starting-stage",
        type=str,
        default="calibration",
        choices=["calibration", "locit", "plotting", "parminator"],
        help="Starting stage in which to start processing (default: %(default)s).",
    )

    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Assume yes on proceed."
    )

    return vars(parser.parse_args())


def param_init(param_dict: dict, msger: MessageBoard):
    if param_dict["root_name"] is None:
        base_name = param_dict["filename"]
    else:
        base_name = param_dict["root_name"]
    base_name_wrds = base_name.split(".")
    if base_name_wrds[-1] == "ms":
        base_name = ".".join(base_name_wrds[:-1])

    param_dict["is_asdm"] = file_is_asdm(param_dict["filename"])
    if param_dict["is_asdm"]:
        param_dict["msname"] = f"{base_name}.ms"
        msger.one_liner("Input is an ASDM, importing it...")
        run_casatask(
            "importasdm",
            {
                "asdm": param_dict["filename"],
                "vis": param_dict["msname"],
                "overwrite": param_dict["overwrite"],
            },
            msger,
        )
    else:
        param_dict["msname"] = param_dict["filename"]

    param_dict["pointing_only_ms"] = f"{base_name}.pnt.ms"
    param_dict["freq_averaged_ms"] = f"{base_name}.avg.ms"
    param_dict["fringefit_caltable"] = f"{base_name}.sbd"
    param_dict["phase_caltable"] = f"{base_name}.pha.gcal"
    param_dict["locit_name"] = f"{base_name}.locit.zarr"
    param_dict["position_name"] = f"{base_name}.position.zarr"

    param_dict["antenna"] = param_dict["antenna"].split(",")
    param_dict["spectral_window"] = param_dict["spectral_window"].split(",")

    if param_dict["scans_to_flag"] is None:
        param_dict["scans_to_flag"] = []
    else:
        param_dict["scans_to_flag"] = param_dict["scans_to_flag"].split(",")

    # Ms data fetching and some consistency checks
    pnt_intent = "CALIBRATE_POINTING#ON_SOURCE"
    msmd = casatools.msmetadata()
    msmd.open(param_dict["msname"])
    ant_names = msmd.antennanames()
    field_names = msmd.fieldnames()
    spw_list = msmd.spwsforintent(pnt_intent)
    nchan = np.unique([msmd.nchan(i_spw) for i_spw in spw_list])
    msmd.done()

    param_dict["n_chan"] = nchan[0]

    error_msgs = []
    if param_dict["refant"] not in ant_names:
        error_msgs.append(f"Chosen refant ({param_dict['refant']}) not present in ms.")
    if param_dict["fringefit_source"] not in field_names:
        error_msgs.append(
            f"Chosen fringefit source ({param_dict['fringefit_source']}) not present in ms."
        )
    if nchan.size != 1:
        error_msgs.append(
            "Spectral windows are not consistent with each other, is this really a pointing ms?"
        )
    if len(error_msgs) > 0:
        raise RuntimeError("\n".join(error_msgs))

    initialization_check(param_dict, "Baseline determination parameters")
    return param_dict


def run_casa_pre_locit_steps(param_dict: dict, msger: MessageBoard):
    run_casatask(
        "split",
        {
            "vis": param_dict["msname"],
            "outputvis": param_dict["pointing_only_ms"],
            "intent": param_dict["intent"],
            "datacolumn": "data",
        },
        msger,
        intended_output=param_dict["pointing_only_ms"],
        overwrite=param_dict["overwrite"],
    )

    if len(param_dict["scans_to_flag"]) > 0:
        run_casatask(
            "flagdata",
            {
                "vis": param_dict["pointing_only_ms"],
                "mode": "manual",
                "scan": ",".join(param_dict["scans_to_flag"]),
                "action": "apply",
                "display": "report",
                "flagbackup": False,
            },
            msger,
        )
        run_casatask(
            "flagmanager",
            {
                "vis": param_dict["pointing_only_ms"],
                "mode": "save",
                "versionname": "baseflags",
            },
            msger,
        )

    fringefit_was_run = run_casatask(
        "fringefit",
        {
            "vis": param_dict["pointing_only_ms"],
            "caltable": param_dict["fringefit_caltable"],
            "field": param_dict["fringefit_source"],
            "solint": "inf",
            "refant": param_dict["refant"],
            "minsnr": 3.0,
            "zerorates": True,
            "globalsolve": True,
            "niter": 100,
        },
        msger,
        intended_output=param_dict["fringefit_caltable"],
        overwrite=param_dict["overwrite"],
    )
    if fringefit_was_run:
        run_casatask(
            "applycal",
            {
                "vis": param_dict["pointing_only_ms"],
                "gaintable": [param_dict["fringefit_caltable"]],
                "interp": ["nearest"],
                "parang": False,
            },
            msger,
        )

    # Now we create a new dataset that is colapsed on the channel axis
    # within each spw, also create a flagversion to store current flag
    # state on the averaged MS
    freq_average_was_run = run_casatask(
        "split",
        {
            "vis": param_dict["pointing_only_ms"],
            "outputvis": param_dict["freq_averaged_ms"],
            "datacolumn": "corrected",
            "keepflags": False,  #
            "width": param_dict["n_chan"],
        },
        msger,
        intended_output=param_dict["freq_averaged_ms"],
        overwrite=param_dict["overwrite"],
    )
    if freq_average_was_run:
        run_casatask(
            "flagmanager",
            {
                "vis": param_dict["freq_averaged_ms"],
                "mode": "save",
                "versionname": "original",
            },
            msger,
        )

    gaincal_was_run = run_casatask(
        "gaincal",
        {
            "vis": param_dict["freq_averaged_ms"],
            "caltable": param_dict["phase_caltable"],
            "solint": "10min",
            "refant": param_dict["refant"],
            "refantmode": "flex",  # Maybe we should use strict for this application?
            "minblperant": 3,
            "minsnr": 3.0,
            "gaintype": "G",  # G is for gain
            "calmode": "p",  # p is for phase
            "solmode": "L1",  # -> least squares
        },
        msger,
        intended_output=param_dict["phase_caltable"],
        overwrite=param_dict["overwrite"],
    )
    if gaincal_was_run:
        run_casatask(
            "applycal",
            {
                "vis": param_dict["freq_averaged_ms"],
                "gaintable": [param_dict["phase_caltable"]],
                "interp": ["nearest"],
                "parang": False,
            },
            msger,
        )

    if not param_dict["assume_yes"]:
        run_casatask(
            "plotms",
            {
                "vis": param_dict["freq_averaged_ms"],
                "xaxis": "time",
                "yaxis": "phase",
                "ydatacolumn": "corrected",
                "field": "*",
                "avgtime": "10",
                "correlation": "RR,LL",
                "coloraxis": "spw",
                "antenna": param_dict["refant"],
                "iteraxis": "baseline",
            },
            msger,
        )
        proceed_check(param_dict, "Are phases clustered around 0 in plotMS?")

    return


def run_astrohack_locit(param_dict: dict, msger: MessageBoard):
    astrohack_param_dict = {
        "cal_table": param_dict["phase_caltable"],
        "locit_name": param_dict["locit_name"],
        "position_name": param_dict["position_name"],
        "ant": param_dict["antenna"],
        "ddi": param_dict["spectral_window"],
        "overwrite": param_dict["overwrite"],
        "fit_kterm": param_dict["fit_kterm"],
        "fit_delay_rate": True,
        "elevation_limit": param_dict["elevation_limit"],
        "polarization": param_dict["polarization"],
        "combine_ddis": param_dict["combination"],
        "parallel": False,
    }
    return


def main():
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.heading("Welcome to the AstroHACK baseline pipeline")

    param_dict = param_init(parse(), msger)
    processing_stage = param_dict["starting_stage"]

    if processing_stage == "calibration":
        run_casa_pre_locit_steps(param_dict, msger)
        processing_stage = "locit"

    if processing_stage == "locit":
        run_astrohack_locit()
        processing_stage = "plotting"

    if processing_stage == "plotting":
        msger.one_liner("PLOTTING WILL COME HERE")
        processing_stage = "parminator"

    if processing_stage == "parminator":
        msger.one_liner("PARMINATOR WILL COME HERE")

    pipeline_end = time.time()
    msger.heading(
        f"Baseline calibration finished in {format_duration(pipeline_end-pipeline_start)}"
    )
    return
