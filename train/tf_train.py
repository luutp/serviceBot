#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
tf_train.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/11/11
'''
# =================================================================================================================
# IMPORT PACKAGES
from __future__ import print_function
import os
import inspect, sys
import argparse
#	File IO
import json
from pathlib import Path
import h5py
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#   AI framework
import tensorflow as tf
import tensorflow.python.keras as keras # tensorflow 2.0
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as KL
import tensorflow.python.keras.engine as KE
import tensorflow.python.keras.models as KM
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
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin
from utils_dir import logger_utils as logger
import config.baseConfig as baseConfig
logger.logging_setup()
# =================================================================================================================
# DEFINES
timenow = datetime.now().strftime('%Y%m%d_%H%M')
config = baseConfig.Config()
config.MODEL['NAME'] = 'TK_MODEL_NAME_2'
config.update()
config.make_fileIO()
PRE_TRAINED_DIR = script_dir / 'pre_trained'

RESNET_URL = 'https://github.com/keras-team/keras-applications/releases/download/resnet'
URL_RESNET50 = RESNET_URL + '/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
URL_RESNET50_NOTOP = RESNET_URL + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
URL_RESNET101 = RESNET_URL + '/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
URL_RESNET101_NOTOP = RESNET_URL + '/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
# =================================================================================================================
@timeit
def download_model_weight(model_url,**kwargs):
    model_dir = PRE_TRAINED_DIR
    model_filename = os.path.split(model_url)[1]
    to_file = get_varargin(kwargs, 'to_file', os.path.join(model_dir, model_filename))
    if os.path.exists(to_file):
        logging.info('File exists: {}. Skip downloading'.format(to_file))
        return 0
    else:
        logging.info('Downloading to: {}'.format(to_file))
        utils.download_url(model_url, to_file)
        
def compile_model(**kwargs):
    pass

def parallel_model(**kwargs):
    pass

def get_last_model(**kwargs):
    pass

def load_model(**kwargs):
    pass

def set_log_dir(**kwargs):
    pass

def main(**kwargs):
    pass


# DEBUG
if __name__ == '__main__':
    download_model_weight(URL_RESNET50_NOTOP)