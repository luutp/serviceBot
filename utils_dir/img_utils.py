#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
img_utils.py
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
import PIL
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
# =================================================================================================================
# TRANSFORM
def img_resize(imgpath, **kwargs):
    width = get_varargin(kwargs, 'width', None)
    height = get_varargin(kwargs, 'height', None)
#    Load image
    img = PIL.Image.open(imgpath)
    logging.info('Image path: {}'.format(imgpath))
    logging.info('Image size: {}'.format(img.size))
#   Compute new width and height
    if width is not None:
        scale_factor = width/img.size[0]
        height = int((float(img.size[1]) * float(scale_factor)))
    else:
        scale_factor = height/img.size[1]
        width = int((float(img.size[0]) * float(scale_factor)))
    logging.info('New size: ({}, {})'.format(width,height))
    img = img.resize((width, height), PIL.Image.ANTIALIAS)
    img.save(imgpath)
    
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    logging.info('Hello!')