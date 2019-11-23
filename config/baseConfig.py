#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
config.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/11/09
'''
# =================================================================================================================
# IMPORT PACKAGES
#%%
from __future__ import print_function
import os
import inspect, sys
import argparse
# File IO
import yaml
import json
from pathlib import Path
from easydict import EasyDict as edict

from tensorflow.python.client import device_lib
# Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
from itertools import chain
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
current_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin
from utils_dir import logger_utils as logger
logger.logging_setup()
# =================================================================================================================
# START
#%%
# Config base class
def model_config():
    MODEL = edict()
    MODEL.NAME = ''
    MODEL.FRAMEWORK = 'tensorflow'
    MODEL.BACKBONE = 'resnet101'
    # Resnet Config
    RESNET = edict()
    RESNET.FILTERS_C2 = [64, 64, 256] # Number of filters in resnet stage 2
    RESNET.FILTERS_C3 = [128, 128, 512]
    RESNET.FILTERS_C4 = [256, 256, 1024]
    RESNET.FILTERS_C5 = [512, 512, 2048]
    MODEL.RESNET = RESNET
    
    # Model URL
    RESNET_URL = 'https://github.com/keras-team/keras-applications/releases/download/resnet'
    MODEL.URL_RESNET50 = RESNET_URL + '/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    MODEL.URL_RESNET50_NOTOP = RESNET_URL + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    MODEL.URL_RESNET101 = RESNET_URL + '/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
    MODEL.URL_RESNET101_NOTOP = RESNET_URL + '/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    return MODEL


def fileIO_config(**kwargs):
    """
     **kwargs:
        project -- str. project name. Default: serviceBot
    Returns:
        FILEIO -- ecict. FileIO configuration
    """
    project_name = get_varargin(kwargs, 'project', 'serviceBot')
    FILEIO = edict()
    # Directories
    FILEIO.PROJECT_DIR = os.path.join(user_dir, project_name)
    FILEIO.LOG_DIR = os.path.join(FILEIO.PROJECT_DIR, 'logs')
    FILEIO.PRE_TRAINED_DIR = os.path.join(FILEIO.PROJECT_DIR, 'train/pre_trained')
    FILEIO.DATASET_DIR = os.path.join(FILEIO.PROJECT_DIR, 'datasets')
    # files
    FILEIO.YAML_CONFIG = os.path.join(current_dir, 'base_config.yaml')
    # Make dirs
    for ipath in [FILEIO.LOG_DIR, FILEIO.DATASET_DIR, FILEIO.PRE_TRAINED_DIR]:
        utils.makedir(ipath)
    
    return FILEIO

def train_config():
    # Train configs
    TRAIN = edict()
    # Get number of GPUs
    def get_nb_gpus(**kwargs):
        verbose = get_varargin(kwargs, 'verbose', False)
        local_devices = device_lib.list_local_devices()
        gpu_list = [x.name for x in local_devices if x.device_type.lower()=='gpu']
        if verbose:
            print(gpu_list)
        return len(gpu_list)
    TRAIN.NB_GPUS = get_nb_gpus()
    TRAIN.USE_GPU = True
    
    TRAIN.OPTIMIZER = 'adam'
    TRAIN.LEARNING_RATE = 0.001 # Learning rate
    TRAIN.LEARNING_MOMENTUM = 0.9
    TRAIN.LEARNING_WEIGHT_DECAY = 0.0001 # Weight decay regularization
    TRAIN.NB_EPOCHS = 10
    TRAIN.BATCH_SIZE = 32
    TRAIN.RESUME = False
    TRAIN.SHUFFLE = True
    TRAIN.PRINT_FREQ = 1 # Frequency to display training process
    
    return TRAIN

def dataset_config():
    DATASET = edict()
    DATASET.NB_CLASSES = 2
    DATASET.IMAGE_SIZE = [224, 224]
    DATASET.IMAGE_CHANNELS = 3
    
    return DATASET
    
#%%
def base_config():
    """
    Base Config class. For custom application, create a sub-class that inherits from this base class
    Arguments:
    object {[type]} -- [description]
    """
    cfg = edict()
    cfg.MODEL = model_config()
    cfg.FILEIO = fileIO_config()
    cfg.TRAIN = train_config()
    return cfg
    
def update_config(to_dict, from_dict):
    pass
    # if str(from_dict) in to_dict.keys():
    #     to_dict.update(from_dict)
    # else:
    #     to_dict. = from_dict
    # for key, val in from_dict.items():
    #     print(key, val)

# Make yaml file
def make_yaml_file(config, yaml_filename):
    logging.info('Saving config yaml file: {}'.format(yaml_filename))
    with open(yaml_filename, 'w') as fid:
        yaml.dump(config, fid)
    logging.info('DONE')
    
# Load yaml file
def load_yaml_file(yaml_filename):
    logging.info('Loading config yaml file: {}'.format(yaml_filename))
    with open(yaml_filename) as fid:
        config = yaml.full_load(fid)
    return config

# cfg = base_config()
# print(cfg.MODEL.FRAMEWORK)
# TEST = edict()
# TEST.NAME = 'merged dict'
# cfg = add_to_config(cfg, TEST)
# # merge_config(cfg, test)
# print(json.dumps(cfg, indent=4))
# =================================================================================================================
# MAIN
#%%
def main(**kwargs):
    pass
# =================================================================================================================
# DEBUG
#%%
if __name__ == '__main__':
    main()
        # yaml_filepath = Path(current_dir) / 'base_config.yaml'
        # main(run_opt = 'loadfile', to_file = yaml_filepath)
