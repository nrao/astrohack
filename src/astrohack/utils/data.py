import copy
import inspect
import json
import datetime
from datetime import date

import toolviper.utils.logger as logger
import numpy as np

import astrohack

from astrohack.utils import compute_average_stokes_visibilities
from astrohack.utils.text import NumpyEncoder


def add_caller_and_version_to_dict(in_dict, direct_call=False):
    if direct_call:
        ipos = 1
    else:
        ipos = 2
    curr_time = datetime.datetime.now()
    local_tz = curr_time.astimezone().tzinfo
    time_str = curr_time.strftime(f"%Y-%m-%d %H:%M:%S {local_tz}")

    in_dict["origin_info"] = {
        "origin": "astrohack",
        "version": astrohack.__version__,
        "creator_function": inspect.stack()[ipos].function,
        "creation_time": time_str,
    }
