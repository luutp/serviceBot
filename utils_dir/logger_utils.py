#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
logger_utils.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/28
'''
# =================================================================================================================
# IMPORT PACKAGES
from __future__ import print_function
import os
import inspect, sys
import time
from datetime import datetime
import logging
import argparse
#  DL framework
from tensorflow.python.client import device_lib
#	Custom packages
import utils
from utils import get_varargin
# =================================================================================================================
# SETUP
#	Working Directories
current_dir = os.getcwd()
user_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
def logging_setup(**kwargs):
    default_logfile = os.path.join(current_dir, 'logging.log')
    log_file = get_varargin(kwargs, 'log_file', default_logfile)
    # 	Logging
    logger = logging.getLogger()
    stream_hdl = logging.StreamHandler(sys.stdout)
    file_hdl = logging.FileHandler(log_file, mode = 'a')
    formatter = logging.Formatter('%(asctime)s | %(filename)s - %(levelname)s - %(message)s', datefmt='%Y%m%d-%I:%M')
    stream_hdl.setFormatter(formatter)
    logger.addHandler(stream_hdl)
    file_hdl.setFormatter(formatter)
    logger.addHandler(file_hdl)
    logger.setLevel(logging.INFO)
    # Only keep one logger
    for h in logger.handlers[:-2]: 
        logger.removeHandler(h)
# =================================================================================================================
# LOG HARDWARE
def logging_hardware(**kwargs):
    logging.info('Hardware information:')
    device_list = device_lib.list_local_devices()
    for device in device_list:
        logging.info('\t Device type: {}'.format(device.device_type))
        logging.info('\t Device name: {}'.format(device.name))
        logging.info('\t Memory Limit: {}(GB)'.format(device.memory_limit/1e9))
        logging.info('\t Physical Info: {}'.format(device.physical_device_desc))


# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    logging_setup()
    logging.info('Hello!')
    # logging_setup()
    # logging_hardware()