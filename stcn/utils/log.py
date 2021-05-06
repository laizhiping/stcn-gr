import torch
import numpy as np
import logging
import time
import os
import json

def get_logger(log_dir, filename=None, verbosity=1, name=None):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if filename is None:
        t = time.strftime("%Y-%m-%d__%H-%M-%S", time.localtime())
        filename = os.path.join(log_dir, t+".log")
    else:
        filename = os.path.join(log_dir, filename)

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(filename)s][%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
