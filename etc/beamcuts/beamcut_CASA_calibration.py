import os
import numpy as np
import casatools
from datetime import datetime, timedelta
from pathlib import Path

lnbr = '\n'
spc=' '

def search(string):
    first = string.find('.')
    if first == -1:
        return -1, -1
    second = string.find('.',first+1)
    return first+1, second+7


def julian_day_to_date(jd):
    # JD 2440587.5 corresponds to 1970-01-01 00:00:00 UTC
    unix_epoch = datetime(1970, 1, 1)
    days_since_unix_epoch = jd - 2440587 #.5
    dt = unix_epoch + timedelta(days=days_since_unix_epoch)
    #return dt.strftime("%Y-%m-%d-Hh-%Mm")
    return dt.strftime("%m-%d-%Y at %Hh%Mm")

def yesno(prompt):
    user_ans = input(f'{prompt} <(Y)es/(N)o>: ').lower()
    if user_ans == 'y' or user_ans == 'yes':
        return True
    elif user_ans == 'n' or user_ans == 'no':
        return False
    else:
        print('Use <yes> or <no>')
        return yesno(prompt)


def is_asdm(filename):
    file_path = Path(f"{filename}/ASDM.xml")
    return file_path.exists()


def asdm_to_ms(asdm_name, overwrite):
    msname = asdm_name + '.ms'
    if os.path.exists(msname) and not overwrite:
        print(msname+' already exists.')
    else:
        importasdm(asdm=asdm_name,
                   vis=msname,
                   createmms=False,
                   ocorr_mode='co',
                   lazy=False,
                   asis='Receiver CalAtmosphere',
                   process_caldevice=True,
                   process_pointing=True,
                   savecmds=True,
                   outfile=msname + '.flagonline.txt',
                   overwrite=False,
                   bdfflags=False,
                   with_pointing_correction=True,
                   applyflags =True)
    return msname


def create_heading(heading, width=60, block_char='#', spacing=1, blocking=3):
    outstr = width*block_char+lnbr
    capo = blocking*block_char + spacing*spc
    coda = capo[::-1]+lnbr
    usable_width = width - 2*spacing - 2*blocking
    heading = heading.strip()
    head_wrds = heading.split()
    capo_len = len(capo)

    def end_line(line, usable_width, coda):
        line_len = len(line) + 1 - capo_len
        spc_to_add = usable_width - line_len
        return spc_to_add*spc + coda

    line = capo
    for wrd in head_wrds:
        wrd_len = len(wrd)
        if wrd_len > usable_width:
            raise ValueError(f'Word {wrd} is larger than the usable width')
        line_len = len(line) + wrd_len + 1 - capo_len
        if line_len > usable_width:
            outstr += line + end_line(line, usable_width, coda)
            line = capo
        else:
            line += spc + wrd
    outstr += line + end_line(line, usable_width, coda)
    outstr += width*block_char+lnbr
    return outstr


class UserInteraction:
    last_use_file = '.beamcut_cal.last'
    user_inp_list = ['filename', 'field', 'refant', 'overwrite']
    sep = '='

    def __init__(self):
        self.filename = None
        self.field = None
        self.refant = None
        self.overwrite = None
        self.last_use_list = None

    def _find_previous_input(self):
        return Path(self.last_use_file).exists()

    def _read_last_use_file(self):
        self.last_use_list = []
        with open(self.last_use_file, 'r') as infile:
            for line in infile:
                self.last_use_list.append(line.strip())

    def _reuse_last(self):
        print('Previous inputs:')
        for line in self.last_use_list:
            print(f'\t{line}')
        return yesno('Re-use previous input?')

    def _init_from_user(self):
        self.filename = input("Enter file name: ")
        self.field = input("Enter field number: ")
        self.refant = input("Enter referece antenna: ")
        self.overwrite = yesno('Re-do calibration if already done?')

    def save_input(self):
        outstr = ''
        for key in self.user_inp_list:
            outstr += f'{key} {self.sep} {getattr(self, key)}\n'
        with open(self.last_use_file, 'w') as outfile:
            outfile.write(outstr)

    def read_input(self):
        if self._find_previous_input():
            self._read_last_use_file()
            if self._reuse_last():
                for line in self.last_use_list:
                    wrds = line.split(self.sep)
                    self.__setattr__(wrds[0], wrds[1])
            else:
                self._init_from_user()
        else:
            self._init_from_user()
        print(self.filename)

    @classmethod
    def perform_beamcut_calibration(cls):

        print(create_heading('Welcome to the beam cut calibration pipeline'))
        my_obj = cls()
        my_obj.read_input()

        print(my_obj.filename)
        if is_asdm(my_obj.filename):
            msname = asdm_to_ms(my_obj.filename, my_obj.overwrite)
        else:
            msname = my_obj.filename

        print(msname)
        # my_obj.save_input()
        # exit()
        # mycal_obj = CalObject(msname, my_obj.field, my_obj.refant, my_obj.overwrite)
        # mycal_obj.calibration_pipeline()
        # mycal_obj.apply_calibration()


class CalObject:

    def __init__(self, msname, field, refant, overwrite, calversion='01'):
        self.msname = msname
        self.refant = refant
        self.overwrite = overwrite
        self.field = field
        # Supposition this will be

        base_cal_name = msname+'.'+calversion+'.'
        self.delay_caltable = base_cal_name+'delay.cal'
        self.bandpass_caltable = 'bandpass.bcal'
        self.gain_caltable = base_cal_name+'gain.cal'

        self._initialize_metadata()
        self._report_init()

    def _initialize_metadata(self):
        # Fetch metadata from ms
        msmd = casatools.msmetadata()
        msmd.open(self.msname)
        cal_scans = msmd.scansforintent('*PHASE*')
        beamcut_scans = msmd.scansforintent('*MAP*ON_SOURCE')
        spw_list = msmd.spwsforintent('*MAP*')
        msmd.done()

        # Convert to comma-separated string
        self.cal_scans = ','.join(map(str, cal_scans))
        self.beamcut_scans = ','.join(map(str, beamcut_scans))

        self.minspw = str(np.min(spw_list))
        self.maxspw = str(np.max(spw_list))
        self.spwrange = self.minspw+'~'+self.maxspw
        self.quacked_spwstr = self.spwrange+':4~60'

        f_dot, l_dot = search(self.msname)
        mod_julian_date = self.msname[f_dot:l_dot]
        julian_date = 2400000.+float(mod_julian_date)
        self.day = julian_day_to_date(julian_date)

    def _report_init(self):
        print('Scans used for calibration:')
        print(self.cal_scans)
        print('Scans used for beamcut:')
        print(self.beamcut_scans)
        print('SPWSs used for beamcuts:')
        print(self.spwrange)
        print('Date obtained:')
        print(self.day)

    def _do_calibration(self, cal_name):
        if os.path.exists(cal_name):
            print(f'{cal_name} exists.')
            if self.overwrite:
                print('\r Overwriting it')
                return True
            else:
                print('\r keeping it')
                return False
        else:
            print(f'{cal_name} does not exist, creating it...')
            return True

    def delay_calibration(self):
        if self._do_calibration(self.delay_caltable):
            gaincal(vis = self.msname,
                    caltable = self.delay_caltable,
                    refant = self.refant,
                    solint = 'inf',
                    spw = self.quacked_spwstr,
                    scan = self.cal_scans,
                    gaintype = 'K')
        return

    def bandpass_calibration(self):
        if self._do_calibration(self.bandpass_caltable):
            bandpass(vis = self.msname,
                     caltable = self.bandpass_caltable,
                     refant = self.refant,
                     solint = '10s',
                     spw = self.quacked_spwstr,
                     solnorm = True,
                     scan = self.cal_scans,
                     gaintable = [self.delay_caltable])
        return

    def gain_calibration(self):
        if self._do_calibration(self.gain_caltable):
            gaincal(vis=self.msname,
                    caltable=self.gain_caltable,
                    refant=self.refant,
                    calmode='ap',
                    solint='inf',
                    spw=self.quacked_spwstr,
                    minsnr=2,
                    minblperant=2,
                    scan=self.cal_scans,
                    gaintable=[self.delay_caltable, self.bandpass_caltable])
        return

    def apply_calibration(self):
        applycal(vis=self.msname,
                 field=self.field,
                 spw=self.quacked_spwstr,
                 applymode='calonly',
                 gaintable=[self.delay_caltable,
                            self.bandpass_caltable,
                            self.gain_caltable]
                 )
        return

    def calibration_pipeline(self):
        self.delay_calibration()
        self.bandpass_calibration()
        self.gain_calibration()
        return


UserInteraction.perform_beamcut_calibration()


