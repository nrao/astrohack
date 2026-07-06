import glob
import inspect
import pathlib
import shutil
from pathlib import Path
import time
from astrohack.utils.text import lnbr, spc, format_duration


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


def run_casatask(
    task_name: str,
    kwargs_dict: dict,
    msger: MessageBoard = None,
    intended_output: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run a casatask and returns True if it has been run, False if it was skipped
    :param task_name: Casatask name
    :param kwargs_dict: Dict containing arguments for casatask
    :param msger: MessageBoard object
    :param intended_output: Possible intended output
    :param overwrite: Overwrite flag (only used when there is an intended output)
    :param verbose: Verbose flag (print messages regarding execution and execution time)
    :return: True when casatask was run, False otherwise
    """
    if intended_output is not None:
        intended_path = pathlib.Path(intended_output)
        if intended_path.exists():
            if overwrite:
                if verbose:
                    msger.one_liner(
                        f"{intended_output} already exists, overwriting it..."
                    )
                if intended_path.is_dir():
                    shutil.rmtree(intended_output)
                    shutil.rmtree(f"{intended_output}.flagversions", ignore_errors=True)
                else:
                    intended_path.unlink(missing_ok=True)
            else:
                if verbose:
                    msger.one_liner(
                        f"{intended_output} already exists, skipping its creation."
                    )
                return False

    if task_name == "plotms":
        import casaplotms

        casatask_func = getattr(casaplotms, "plotms")
    else:
        import casatasks

        casatask_func = getattr(casatasks, task_name)

    if verbose:
        msger.one_liner(f"Running {task_name}...")
    task_start_time = time.time()
    casatask_func(**kwargs_dict)
    task_end_time = time.time()
    if verbose:
        msger.one_liner(
            f"{task_name} finished in {format_duration(task_end_time - task_start_time)}."
        )
    return True


def parse_list_or_all(parameter_value: str):
    if parameter_value == "all":
        return "all"
    else:
        return parameter_value.split(",")


def print_dict_simple(the_dict, ident=4):
    key_len = 0
    for key in the_dict.keys():
        if len(key) > key_len:
            key_len = len(key)

    for key, value in the_dict.items():
        print(f"{ident*' '}{key:{key_len}s} => {value}")


def initialization_check(param_dict: dict, title: str):
    print()
    print(f"{title}:")
    print_dict_simple(param_dict)
    print()
    proceed_check(param_dict)
    print()


def proceed_check(param_dict: dict, prompt: str = "Proceed?"):
    if not param_dict["assume_yes"]:
        if not yesno(prompt):
            exit(0)


def list_input_tooltip(example):
    return f"for a list use comma separated values with no spaces, e.g.: '{example}'"


def created_filtered_kwargs_dict(param_dict: dict, function):
    valid_kwarg_keys = inspect.signature(function).parameters
    filtered_dict = {
        key: value for key, value in param_dict.items() if key in valid_kwarg_keys
    }
    return filtered_dict


def run_astrohack_function(
    param_dict: dict, function, msger: MessageBoard, verbose: bool = True
):
    function_name = function.__name__
    try:
        if verbose:
            msger.one_liner(f"Running {function_name}...")
        function_start_time = time.time()
        function(**created_filtered_kwargs_dict(param_dict, function))
        function_end_time = time.time()
        if verbose:
            msger.one_liner(
                f"{function_name} finished in {format_duration(function_end_time - function_start_time)}."
            )
        return True, None
    except Exception as the_exception:
        return False, the_exception
