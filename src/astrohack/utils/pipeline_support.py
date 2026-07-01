import shutil
from pathlib import Path
import time
from astrohack.utils.text import lnbr, spc


def yesno(prompt):
    user_ans = input(f"{prompt} <(Y)es/(N)o>: ").lower()
    if user_ans == "y" or user_ans == "yes":
        return True
    elif user_ans == "n" or user_ans == "no":
        return False
    else:
        print("Use <yes> or <no>")
        return yesno(prompt)


def file_is_asdm(filename):
    file_path = Path(f"{filename}/ASDM.xml")
    return file_path.exists()


def run_casatask(task_name: str, kwargs_dict: dict, msger):
    import casatasks

    casatask_func = getattr(casatasks, task_name)
    msger.one_liner(f"Running {task_name}...")
    task_start_time = time.time()
    casatask_func(**kwargs_dict)
    task_end_time = time.time()
    msger.one_liner(f"{task_name} finished in {task_end_time - task_start_time:.2f} s")


class MessageBoard:

    def __init__(self, width=None, block_char="#", spacing=1, blocking=3):
        if width is None:
            term_size = shutil.get_terminal_size((80, 20))
            self.width = term_size.columns
        else:
            self.width = width
        self.block_char = block_char
        self.spacing = (spacing,)
        self.blocking = blocking

        self.capo = blocking * block_char + spacing * spc
        self.coda = self.capo[::-1] + lnbr
        self.usable_width = self.width - 2 * spacing - 2 * blocking
        self.block_line = self.width * self.block_char + lnbr
        self.block_len = len(self.capo)

    def _end_line(self, line):
        line_len = len(line) + 1 - self.block_len
        spc_to_add = (self.usable_width - line_len - self.blocking) * spc
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
                outstr += self._end_line(line)
                line = wrd
            else:
                line += spc + wrd
        outstr += self._end_line(line)
        outstr += self.block_line
        print(outstr)
        return outstr

    def one_liner(self, msg):
        outstr = self.capo + msg + lnbr
        print(outstr)
        return outstr

    def done(self):
        return self.one_liner("Done!")


def print_dict_simple(the_dict, ident=4):
    key_len = 0
    for key in the_dict.keys():
        if len(key) > key_len:
            key_len = len(key)

    for key, value in the_dict.items():
        print(f"{ident*' '}{key:{key_len}s} => {value}")


def initialization_check(param_dict: dict, title: str):
    print(f"{title}:")
    print_dict_simple(param_dict)
    print()
    if not param_dict["assume_yes"]:
        if not yesno("Proceed?"):
            exit(0)
    print()
