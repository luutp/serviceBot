#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
image_captioning.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/12/03
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
from pathlib import Path
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#   COCO
# from pycocotools.coco import COCO
from datasets.coco.coco import COCO
#	Visualization Packages
import matplotlib.pyplot as plt
import skimage.io as skio
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
script_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin
import utils_dir.utils as utils
from utils_dir import logger_utils as logger
from config.baseConfig import base_config
logger.logging_setup()
cfg = base_config()
cfg.DATASET.CocoCaption = '/media/phatluu/Ubuntu_D/datasets/coco/annotations/captions_train2014.json'
# =================================================================================================================
# MAIN
def main(**kwargs):
    coco = COCO(cfg.DATASET.CocoCaption)    
    coco.filter_by_cap_len(20)
# main()
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()

# %%
