import argparse
import os
import shutil

import numpy as np
import casatools
from pathlib import Path
from casatasks import importasdm, gaincal, bandpass, applycal
from astrohack.utils.pipeline_support import (
    MessageBoard,
    initialization_check,
    file_is_asdm,
)


def parse():
    parser = argparse.ArgumentParser(description="Beam cut CASA calibration pipeline")

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
        "--beamcut-field",
        default=None,
        type=str,
        help="Field Id or name of the beam cut data (default is to determine it from data)",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing calibration files",
    )

    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Assume yes on proceed."
    )

    parser.add_argument(
        "-q",
        "--quack-nchan",
        default=4,
        type=int,
        help="Number of channels to quack at the edge of the spectral window (default is %(default)s)",
    )

    param_dict = vars(parser.parse_args())
    return param_dict


class CalObject:

    def __init__(self, param_dict: dict, msger: MessageBoard):
        self.filename = param_dict["filename"]
        self.refant = param_dict["refant"]
        self.overwrite = param_dict["overwrite"]
        self.is_asdm = file_is_asdm(self.filename)

        if param_dict["root_name"] is None:
            base_cal_name = f"{self.filename}."
        else:
            base_cal_name = param_dict["root_name"] + "."

        self.delay_caltable = base_cal_name + "delay.cal"
        self.bandpass_caltable = base_cal_name + "bandpass.cal"
        self.gain_caltable = base_cal_name + "gain.cal"

        if self.is_asdm:
            self.msname = f"{self.filename}.ms"
            msger.one_liner("Input is an SDM running importasdm...")
            self.asdm_to_ms()
            msger.done()
        else:
            self.msname = self.filename

        self._initialize_metadata(param_dict["quack_nchan"])
        param_dict.update(
            {
                key: value
                for key, value in vars(self).items()
                if not key.startswith("__")
            }
        )
        initialization_check(
            param_dict,
            "CASA calibration parameters",
        )
        self.msger = msger

    def asdm_to_ms(self):
        if os.path.exists(self.msname) and self.overwrite:
            self.msger.heading(f"Removing previous ms file ({self.msname})")
            shutil.rmtree(self.msname)

        importasdm(
            asdm=self.filename,
            vis=self.msname,
            createmms=False,
            ocorr_mode="co",
            lazy=False,
            asis="Receiver CalAtmosphere",
            process_caldevice=True,
            process_pointing=True,
            savecmds=True,
            outfile=f"{self.filename}.flagonline.txt",
            bdfflags=False,
            with_pointing_correction=True,
            applyflags=True,
            overwrite=False,
        )
        return

    def _initialize_metadata(self, quack_nchan):
        # Fetch metadata from ms
        msmd = casatools.msmetadata()
        msmd.open(self.msname)
        cal_scans = msmd.scansforintent("*PHASE*")
        beamcut_scans = msmd.scansforintent("*MAP*ON_SOURCE")
        spw_list = msmd.spwsforintent("*MAP*")
        beamcut_fields = np.unique(msmd.fieldsforscans(beamcut_scans))
        nchan = np.unique([msmd.nchan(i_spw) for i_spw in spw_list])
        msmd.done()

        if beamcut_fields.size > 1:
            raise RuntimeError("More than 1 beam cut field, try splitting the ms")
        if nchan.size > 1:
            raise RuntimeError(
                "Spectral windows have different nchans, don't know how to proceed automatically"
            )

        fchan = quack_nchan
        lchan = nchan[0] - quack_nchan
        # Convert to comma-separated string
        self.calibration_scans = ",".join(map(str, cal_scans))
        self.beamcut_scans = ",".join(map(str, beamcut_scans))
        self.beamcut_field = beamcut_fields[0]

        minspw = f"{round(np.min(spw_list)):d}"
        maxspw = f"{round(np.max(spw_list)):d}"
        spwrange = f"{minspw}~{maxspw}"
        self.quacked_spw_selection = f"{spwrange}:{fchan}~{lchan}"

    def _do_calibration(self, cal_name):
        if os.path.exists(cal_name):
            if self.overwrite:
                self.msger.one_liner(f"{cal_name} exists, overwriting.")
                return True
            else:
                self.msger.one_liner(f"{cal_name} exists, keeping it.")
                return False
        else:
            self.msger.one_liner(f"{cal_name} does not exist, creating it...")
            return True

    def delay_calibration(self):
        self.msger.one_liner("Delay calibration...")
        if self._do_calibration(self.delay_caltable):
            gaincal(
                vis=self.msname,
                caltable=self.delay_caltable,
                refant=self.refant,
                solint="inf",
                spw=self.quacked_spw_selection,
                scan=self.calibration_scans,
                gaintype="K",
            )
            self.msger.done()
        else:
            self.msger.one_liner("Skipping delay calibration...")
        return

    def bandpass_calibration(self):
        self.msger.one_liner("Bandpass calibration...")
        if self._do_calibration(self.bandpass_caltable):
            bandpass(
                vis=self.msname,
                caltable=self.bandpass_caltable,
                refant=self.refant,
                solint="10s",
                spw=self.quacked_spw_selection,
                solnorm=True,
                scan=self.calibration_scans,
                gaintable=[self.delay_caltable],
            )
            self.msger.done()
        else:
            self.msger.one_liner("Skipping bandpass calibration...")
        return

    def gain_calibration(self):
        self.msger.one_liner("Gain calibration...")
        if self._do_calibration(self.gain_caltable):
            gaincal(
                vis=self.msname,
                caltable=self.gain_caltable,
                refant=self.refant,
                calmode="ap",
                solint="inf",
                spw=self.quacked_spw_selection,
                minsnr=2,
                minblperant=2,
                scan=self.calibration_scans,
                gaintable=[self.delay_caltable, self.bandpass_caltable],
            )
            self.msger.done()
        else:
            self.msger.one_liner("Skipping gain calibration...")
        return

    def apply_calibration(self):
        self.msger.one_liner("Applying calibration...")
        applycal(
            vis=self.msname,
            field=f"{self.beamcut_field}",
            spw=self.quacked_spw_selection,
            applymode="calonly",
            gaintable=[self.delay_caltable, self.bandpass_caltable, self.gain_caltable],
        )
        self.msger.done()
        return

    def calibration_pipeline(self):
        self.delay_calibration()
        self.bandpass_calibration()
        self.gain_calibration()
        return


def main():
    msger = MessageBoard()
    msger.heading("Welcome to CASA beam cut calibration pipeline")
    cal_param_dict = parse()

    mycal_obj = CalObject(cal_param_dict, msger)
    mycal_obj.calibration_pipeline()
    mycal_obj.apply_calibration()

    msger.heading("Beamcut calibration complete!")
