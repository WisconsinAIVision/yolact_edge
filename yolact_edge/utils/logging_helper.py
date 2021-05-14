import logging
import sys
from termcolor import colored
import os

"""
Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py
"""

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def log_once(obj, msg_key, name, message, level=logging.WARNING, *args, **kwargs):
    if getattr(obj, "logged_" + msg_key, False):
        return
    
    setattr(obj, "logged_" + msg_key, True)
    logger = logging.getLogger(name)
    logger.log(level, message, *args, **kwargs)


def setup_logger(name="yolact", output=None, distributed_rank=0, abbrev_name=None, logging_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "yolact" if name == "yolact" else name

    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging_level)

        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )

        ch.setFormatter(formatter)
        logger.addHandler(ch)

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(open(filename, "a"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger
