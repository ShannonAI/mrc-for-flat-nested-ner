#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# file: get_logger.py


import logging


def logger_to_file(logfile_path):
    format = '%(asctime)s - %(message)s'
    logging.basicConfig(format=format, filename=logfile_path, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger



