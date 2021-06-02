"""
This file contains some functions and classes which can be useful in many projects.
"""

import os
import sys
import torch
import random
import logging
import traceback
import numpy as np


class Table:
    """This class is used to pretty-print tables of data, similar to https://pypi.org/project/tabulate/
    It is not used during training.
    TOREMOVE
    """

    def __init__(self, header=None):
        self.header = header;
        self.rows = []

    def add(self, row):
        if self.header != None or len(self.rows) != 0:
            assert len(row) == self.num_cols(), f"Different len: {len(row)} {self.num_cols()}"
        self.rows.append(row)

    def num_cols(self):
        if self.header != None:
            return len(self.header)
        else:
            return len(self.rows[0])

    def num_rows(self):
        return len(self.rows)

    def show(self, line_number=True, sep="   "):
        def _s_(obj):
            return str(obj).replace("\n", "")

        if self.header == None and len(self.rows) == 0: return "Tabella vuota"
        if self.header != None:
            all_rows = [self.header] + self.rows
        else:
            all_rows = self.rows
        col_widths = [max([len(_s_(all_rows[r][c])) for r in range(len(all_rows))]) for c in range(self.num_cols())]
        table = [[f"{_s_(all_rows[r][c]):<{col_widths[c]}}" for c in range(self.num_cols())] for r in
                 range(len(all_rows))]
        if line_number:
            table = "\n".join([f"{i:2d}  " + sep.join(row) for i, row in enumerate(table)])
        else:
            table = "\n".join([sep.join(row) for row in table])
        return table

    def __repr__(self):
        print(self.show()); return ""

    def sort(self, col_index):
        self.rows = sorted(self.rows, key=lambda x: x[col_index])


def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_folder, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('shapely').disabled = True
    logging.getLogger('shapely.geometry').disabled = True
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.INFO)  # turn off logging tag for some images

    if info_filename != None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename != None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console != None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = exception_handler

