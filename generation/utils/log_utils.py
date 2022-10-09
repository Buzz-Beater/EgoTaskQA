"""

    Created on 2021/12/20

    @author: Baoxiong Jia

"""

import logging
import os
import tqdm
from pathlib import Path


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# _LOG_FMT = "%(asctime)s - %(levelname)s -   %(message)s"
_LOG_FMT = "%(message)s"
_DATE_FMT = "%m/%d/%Y %H:%M:%S"
LOGGER = logging.getLogger("")
LOGGER.setLevel(logging.INFO)
th = TqdmLoggingHandler()
th.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
LOGGER.addHandler(th)


def add_log_to_file(log_path):
    """Add file handler to store log to files"""
    if Path(log_path).exists():
        os.remove(log_path)
    fh = logging.FileHandler(str(log_path))
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


class AverageMeter(object):
    """Average meter for basic numbers"""

    def __init__(self, keys=None):
        if keys is not None:
            self.keys = keys + ["avg"]
        else:
            self.keys = ["avg"]
        self.reset()
        self.type = type

    def reset(self):
        self.dict = {key: 0. for key in self.keys}
        for key in self.keys:
            self.dict[key] = {"val": 0., "total": 0., "score": 0.}
        self.epsilon = 1e-20

    def get(self, key):
        return self.dict[key]

    def update(self, val, total, key="avg"):
        self.dict[key]["val"] += val
        self.dict[key]["total"] += total
        self.dict[key]["score"] = self.dict[key]["val"] / (self.dict[key]["total"] + self.epsilon)

    def gen_stats(self):
        log_strs = [f"{k}: {v['score']:.4f}" for k, v in self.dict.items()]
        log_str = ",".join(log_strs)
        return log_str