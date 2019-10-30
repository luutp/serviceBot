#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
fileIO_utils.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/30
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
from utils import get_varargin, timeit
# =================================================================================================================
# SETUP
#	Working Directories
current_dir = os.getcwd()
user_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
# =================================================================================================================
@timeit
def unzip_file(filename, **kwargs):
    '''
    unzip file
    '''
    output_dir = get_varargin(kwargs, 'output', os.path.dirname(filename))
    del_zip = get_varargin(kwargs,'remove', True)
    import zipfile
    with zipfile.ZipFile(filename, 'r') as fid:
        fid.extractall(output_dir)
    if del_zip is True:
        os.remove(filename)
# =================================================================================================================
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    logging.info('Hello!')