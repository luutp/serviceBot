#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
template_py.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/01/20
'''
# =================================================================================================================
# IMPORT PACKAGES
#%%
from __future__ import print_function
import os
import inspect, sys
import argparse
#	File IO
import json
import yaml
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#	Visualization Packages
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2
import seaborn as sns
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
project_dir = os.path.join(user_dir, 'serviceBot')
if not (project_dir in sys.path):
    print('sys.append: {}'.format(project_dir))
    sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin, verbose
import  utils_dir.utils as utils
from utils_dir import logger_utils as logger_utils
logger = logger_utils.htmlLogger(log_file = os.path.join(project_dir, '{}_logging.html'\
    .format(datetime.now().strftime('%y%m%d'))), mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))

def plt_config():
    plt.rcParams['figure.figsize'] = (8,8)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 1.75
    plt.rcParams['lines.color'] = 'k'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
plt_config()
#%%
# =================================================================================================================
# FUNCTIONS and CLASSES
class classname(object):
    # Initialize Class
    def __init__(self, prop1):
        self._prop1 = prop1
# =================================================================================================================
    @property
    def prop1(self):
        return self._prop1
    @prop1.setter
    def prop1(self, inputVal):
        self._prop1 = inputVal
# =================================================================================================================
# METHODS
    def func1(self, **kwargs):
        pass
# =================================================================================================================
# STATIC METHODS
    @staticmethod
    def function1(**kwargs):
        pass
# =================================================================================================================
# MAIN
def main(**kwargs):
    pass
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()