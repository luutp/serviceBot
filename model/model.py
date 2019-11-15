#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
model.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/11/12
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
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#   AI Framework
import tensorflow as tf
import tensorflow.python.keras as keras # tensorflow 2.0
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as KL
import tensorflow.python.keras.engine as KE
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape
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
from utils_dir import utils
from utils_dir.utils import timeit, get_varargin
from utils_dir import logger_utils as logger
from config.baseConfig import Config
logger.logging_setup()
# DEFINE
config = Config()
# =================================================================================================================
# MODEL UTILS
def export_keras_model_structure(keras_model, filepath, **kwargs):
    model = keras_model
    model_structure = model.to_json()
    # Save model architecture in JSON file
    with open(filepath, 'w') as fid:
        fid.write(model_structure)
# 


# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    # model = ResNet50(include_top=True)
    download_model_weight(URL_RESNET50_NOTOP)
    # model.summary()
    # export_keras_model_structure(model, 'keras_model_structure.json')