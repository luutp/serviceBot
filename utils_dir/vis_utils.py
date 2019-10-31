#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
vis_utils.py
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
#	Data Analytics
import pandas as pd
import numpy as np
#	Visualization Packages
import matplotlib.pyplot as plt
#	Custom packages
import utils
from utils import get_varargin
# =================================================================================================================
# SETUP
#	Working Directories
current_dir = os.getcwd()
user_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
# 	Logging
logger = logging.getLogger()
stream_hdl = logging.StreamHandler(sys.stdout)
file_hdl = logging.FileHandler(os.path.join(current_dir,'logging.log'), mode = 'a')
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
# DEBUG
if __name__ == '__main__':
    logging.info('Hello!')