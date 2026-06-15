import os
import shutil

import numpy as np
import casatools
from pathlib import Path

lnbr = "\n"
spc = " "


def yesno(prompt):
    user_ans = input(f"{prompt} <(Y)es/(N)o>: ").lower()
    if user_ans == "y" or user_ans == "yes":
        return True
    elif user_ans == "n" or user_ans == "no":
        return False
    else:
        print("Use <yes> or <no>")
        return yesno(prompt)


class MessageBoard:

    def __init__(self, width=60, block_char="#", spacing=1, blocking=3):
        self.width = width
        self.block_char = block_char
        self.spacing = (spacing,)
        self.blocking = blocking

        self.capo = blocking * block_char + spacing * spc
        self.coda = self.capo[::-1] + lnbr
        self.usable_width = width - 2 * spacing - 2 * blocking
        self.block_line = self.width * self.block_char + lnbr
        self.block_len = len(self.capo)

    def end_line(self, line, centered=True):
        line_len = len(line) + 1 - self.block_len
        spc_to_add = self.usable_width - line_len
        if centered:
            out_line = self.capo + line + self.coda
        else:
            out_line = self.capo + line + spc_to_add + self.coda
        return out_line

    def heading(self, user_msg):
        outstr = ""
        outstr += self.block_line
        head_wrds = user_msg.split()

        line = ""
        for wrd in head_wrds:
            wrd_len = len(wrd)
            if wrd_len > self.usable_width:
                raise ValueError(f"Word {wrd} is larger than the usable self.width")
            line_len = len(line) + wrd_len + 1
            if line_len > self.usable_width:
                outstr += self.end_line(line)
                line = wrd
            else:
                line += spc + wrd
        outstr += self.end_line(line)
        outstr += self.block_line
        return outstr

    def one_liner(self, msg):
        if len(msg) > self.usable_width:
            raise ValueError("Message is larger than usable width")
        return self.end_line(msg)

    def done(self):
        return self.one_liner("Done!")


class UserInteraction:
    last_use_file = ".beamcut_cal.last"
    user_inp_list = [
        "filename",
        "field",
        "refant",
        "overwrite",
        "confirmation_before_start",
    ]
    sep = "="

    def __init__(self):
        self.filename = None
        self.field = None
        self.refant = None
        self.overwrite = None
        self.last_use_list = None
        self.confirmation_before_start = None

    def _find_previous_input(self):
        return Path(self.last_use_file).exists()

    def _read_last_use_file(self):
        self.last_use_list = []
        with open(self.last_use_file, "r") as infile:
            for line in infile:
                self.last_use_list.append(line.strip())

    def _reuse_last(self):
        print("Previous inputs:")
        for line in self.last_use_list:
            print(f"\t{line}")
        ans = yesno("Re-use previous input?")
        print()
        return ans

    def _init_from_user(self):
        self.filename = input("Enter MS/ASDM file name: ")
        self.field = input("Enter beam cut field number: ")
        self.refant = input("Enter reference antenna for calibration: ")
        self.overwrite = yesno("Re-do calibration if already done?")
        self.confirmation_before_start = yesno(
            "Confirm info before starting calibration?"
        )

    def save_input(self):
        outstr = ""
        for key in self.user_inp_list:
            outstr += f"{key} {self.sep} {getattr(self, key)}\n"
        with open(self.last_use_file, "w") as outfile:
            outfile.write(outstr)

    def read_input(self):
        if self._find_previous_input():
            self._read_last_use_file()
            if self._reuse_last():
                for line in self.last_use_list:
                    wrds = line.split(self.sep)
                    key = wrds[0].strip()
                    value = wrds[1].strip()
                    setattr(self, key, value)
            else:
                self._init_from_user()
        else:
            self._init_from_user()

    @classmethod
    def perform_beamcut_calibration(cls):
        msger = MessageBoard()
        print(msger.heading("Welcome to the beam cut calibration pipeline"))
        my_obj = cls()
        my_obj.read_input()
        print()

        mycal_obj = CalObject(
            my_obj.filename, my_obj.field, my_obj.refant, my_obj.overwrite, msger
        )
        if my_obj.confirmation_before_start:
            proceed = yesno("Proceed with calibration?")
        else:
            proceed = True
        print()

        if proceed:
            my_obj.save_input()
            mycal_obj.calibration_pipeline()
            mycal_obj.apply_calibration()

        print(msger.heading("All Done!"))


class CalObject:

    def __init__(
        self, msname, field, refant, overwrite, msger, first_chan=4, last_chan=60
    ):
        self.msname = msname
        self.refant = refant
        self.overwrite = bool(overwrite)
        self.field = field
        self.msger = msger
        self.fchan = first_chan
        self.lchan = last_chan

        base_cal_name = msname + "."
        self.delay_caltable = base_cal_name + "delay.cal"
        self.bandpass_caltable = base_cal_name + "bandpass.cal"
        self.gain_caltable = base_cal_name + "gain.cal"

        if self._is_asdm():
            print(self.msger.one_liner("Input is an SDM running importasdm..."))
            self.asdm_to_ms()
            print(self.msger.done())

        self._initialize_metadata()
        self._report_init()

    def _is_asdm(self):
        file_path = Path(f"{self.msname}/ASDM.xml")
        return file_path.exists()

    def asdm_to_ms(self):
        msname = self.msname + ".ms"
        if os.path.exists(msname) and self.overwrite:
            print(self.msger.heading("Removing old file"))
            shutil.rmtree(msname)

        importasdm(
            asdm=self.msname,
            vis=msname,
            createmms=False,
            ocorr_mode="co",
            lazy=False,
            asis="Receiver CalAtmosphere",
            process_caldevice=True,
            process_pointing=True,
            savecmds=True,
            outfile=msname + ".flagonline.txt",
            bdfflags=False,
            with_pointing_correction=True,
            applyflags=True,
            overwrite=False,
        )
        self.msname = msname
        return

    def _initialize_metadata(self):
        # Fetch metadata from ms
        msmd = casatools.msmetadata()
        msmd.open(self.msname)
        cal_scans = msmd.scansforintent("*PHASE*")
        beamcut_scans = msmd.scansforintent("*MAP*ON_SOURCE")
        spw_list = msmd.spwsforintent("*MAP*")
        msmd.done()

        # Convert to comma-separated string
        self.cal_scans = ",".join(map(str, cal_scans))
        self.beamcut_scans = ",".join(map(str, beamcut_scans))

        self.minspw = str(np.min(spw_list))
        self.maxspw = str(np.max(spw_list))
        self.spwrange = self.minspw + "~" + self.maxspw
        self.quacked_spwstr = self.spwrange + f":{self.fchan}~{self.lchan}"

    def _report_init(self):
        print("Scans used for calibration:")
        print(self.cal_scans)
        print()
        print("Scans used for beamcut:")
        print(self.beamcut_scans)
        print()
        print("SPWSs used for beamcuts:")
        print(self.spwrange)
        print()

    def _do_calibration(self, cal_name):
        if os.path.exists(cal_name):
            print(f"{cal_name} exists.")
            if self.overwrite:
                print(f"{cal_name} exists, overwriting.")
                return True
            else:
                print(f"{cal_name} exists, keeping it.")
                return False
        else:
            print(f"{cal_name} does not exist, creating it...")
            return True

    def delay_calibration(self):
        print(self.msger.one_liner("Delay calibration..."))
        if self._do_calibration(self.delay_caltable):
            gaincal(
                vis=self.msname,
                caltable=self.delay_caltable,
                refant=self.refant,
                solint="inf",
                spw=self.quacked_spwstr,
                scan=self.cal_scans,
                gaintype="K",
            )
            print(self.msger.done())
        else:
            print(self.msger.one_liner("Skipping delay calibration..."))
        return

    def bandpass_calibration(self):
        print(self.msger.one_liner("Bandpass calibration..."))
        if self._do_calibration(self.bandpass_caltable):
            bandpass(
                vis=self.msname,
                caltable=self.bandpass_caltable,
                refant=self.refant,
                solint="10s",
                spw=self.quacked_spwstr,
                solnorm=True,
                scan=self.cal_scans,
                gaintable=[self.delay_caltable],
            )
            print(self.msger.done())
        else:
            print(self.msger.one_liner("Skipping bandpass calibration..."))
        return

    def gain_calibration(self):
        print(self.msger.one_liner("Gain calibration..."))
        if self._do_calibration(self.gain_caltable):
            gaincal(
                vis=self.msname,
                caltable=self.gain_caltable,
                refant=self.refant,
                calmode="ap",
                solint="inf",
                spw=self.quacked_spwstr,
                minsnr=2,
                minblperant=2,
                scan=self.cal_scans,
                gaintable=[self.delay_caltable, self.bandpass_caltable],
            )
            print(self.msger.done())
        else:
            print(self.msger.one_liner("Skipping gain calibration..."))
        return

    def apply_calibration(self):
        print(self.msger.one_liner("Applying calibration..."))
        applycal(
            vis=self.msname,
            field=self.field,
            spw=self.quacked_spwstr,
            applymode="calonly",
            gaintable=[self.delay_caltable, self.bandpass_caltable, self.gain_caltable],
        )
        print(self.msger.done())
        return

    def calibration_pipeline(self):
        self.delay_calibration()
        self.bandpass_calibration()
        self.gain_calibration()
        return


def main():
    UserInteraction.perform_beamcut_calibration()
