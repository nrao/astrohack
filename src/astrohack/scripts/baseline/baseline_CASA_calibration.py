import argparse
import time

import casatools
import numpy as np
from pathlib import Path
import shutil

from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    file_is_asdm,
    run_casatask,
    proceed_check,
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
        "-s",
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

    initialization_check(param_dict, "Baseline CASA calibration parameters")
    return param_dict


def run_casa_pre_locit_steps(param_dict: dict, msger: MessageBoard):
    if Path(param_dict["pointing_only_ms"]).is_dir() and not param_dict["overwrite"]:
        msger.one_liner(
            f"{param_dict['pointing_only_ms']} already exists, skipping splitting."
        )
    else:
        run_casatask(
            "split",
            {
                "vis": param_dict["msname"],
                "outputvis": param_dict["pointing_only_ms"],
                "intent": param_dict["intent"],
                "datacolumn": "data",
            },
            msger,
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

    if Path(param_dict["fringefit_caltable"]).is_dir() and not param_dict["overwrite"]:
        msger.one_liner(
            f"{param_dict['fringefit_caltable']} already exists, skipping fringefit."
        )
    else:
        run_casatask(
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
        )
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
    if Path(param_dict["freq_averaged_ms"]).is_dir() and not param_dict["overwrite"]:
        msger.one_liner("Frequency averaged ms already exists, skipping its creation.")
    else:
        run_casatask(
            "split",
            {
                "vis": param_dict["pointing_only_ms"],
                "outputvis": param_dict["freq_averaged_ms"],
                "datacolumn": "corrected",
                "keepflags": False,  #
                "width": param_dict["n_chan"],
            },
            msger,
        )
        run_casatask(
            "flagmanager",
            {
                "vis": param_dict["freq_averaged_ms"],
                "mode": "save",
                "versionname": "original",
            },
            msger,
        )

    if Path(param_dict["phase_caltable"]).is_dir() and not param_dict["overwrite"]:
        run_casatask(
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
        )
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
                "yaxis": "phase",
                "ydatacolumn": "data",
                "field": "*",
                "avgtime": "10",
                "correlation": "RR,LL",
                "coloraxis": "spw",
                "antenna": param_dict["refant"],
                "iteraxis": "baseline",
            },
            msger,
        )
        print("Do phases look good in plot ms?")
        proceed_check(param_dict)

    return


def main():
    pipeline_start = time.time()
    msger = MessageBoard()
    print()
    msger.heading("Welcome to the astrohack baseline pipeline")

    param_dict = param_init(parse(), msger)
    processing_stage = param_dict["starting_stage"]

    if processing_stage == "calibration":
        run_casa_pre_locit_steps(param_dict, msger)
        processing_stage = "locit"

    if processing_stage == "locit":
        msger.one_liner("LOCIT WILL COME HERE")
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
