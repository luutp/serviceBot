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
from pathlib import Path
from easydict import EasyDict as edict
# Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
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
# Config base class
#%%
class Config(object):
    """
    Base Config class. For custom application, create a sub-class that inherits from this base class
    Arguments:
    object {[type]} -- [description]
    """
    # Model configs
    MODEL = edict()
    MODEL.NAME = ''
    MODEL.FRAMEWORK = 'tensorflow'
    MODEL.BACKBONE = 'resnet101'
    # Resnet config
    RESNET = dict()
    RESNET['FILTERS_C2'] = [64, 64, 256] # Number of filters in resnet stage 2
    RESNET['FILTERS_C3'] = [128, 128, 512]
    RESNET['FILTERS_C4'] = [256, 256, 1024]
    RESNET['FILTERS_C5'] = [512, 512, 2048]
    MODEL['RESNET'] = RESNET
    
    # Project path configs
    FILEIO = dict()
    FILEIO['PROJECT_DIR'] = os.path.abspath(project_dir)
    FILEIO['CONFIG_YAML'] = 'untitled.yaml'
    FILEIO['LOG_DIR'] = os.path.join(FILEIO['PROJECT_DIR'], 'logModels')
    # Hardware configs
    HARDWARE = dict()
    HARDWARE['GPU_COUNT'] = 1
    
    # Datasets config
    DATASET = dict()
    DATASET['NB_CLASSES'] = 2
    DATASET['IMAGE_SIZE'] = [224, 224]
    DATASET['IMAGE_CHANNELS'] = 3
    
    # Train configs
    TRAIN = dict()
    TRAIN['OPTIMIZER'] = 'adam'
    TRAIN['LEARNING_RATE'] = 0.001 # Learning rate
    TRAIN['LEARNING_MOMENTUM'] = 0.9
    TRAIN['LEARNING_WEIGHT_DECAY'] = 0.0001 # Weight decay regularization
    
    TRAIN['NB_EPOCH'] = 10
    TRAIN['BATCH_SIZE'] = 32
    TRAIN['RESUME'] = False
    TRAIN['SHUFFLE'] = True
    TRAIN['PRINT_FREQ'] = 100 # Frequency to display training process
    # Test configs
    # config
    config = dict()
    config['HARDWARE'] = HARDWARE
    config['DATASET'] = DATASET
    config['MODEL'] = MODEL
    config['TRAIN'] = TRAIN
    config['FILEIO'] = FILEIO
    # ==========================================================================================================
    def __init__(self):
        pass
    # =================================================================================================================
    # Methods 
    def update(self, **kwargs):
        _yaml_filepath = os.path.join(self.FILEIO['PROJECT_DIR'],
                                      'config/{}_config.yaml'.format(self.MODEL['NAME'].lower()))
        self.FILEIO['CONFIG_YAML'] = _yaml_filepath
        self.config['FILEIO'] = self.FILEIO
        
    def make_fileIO(self, **kwargs):
        # Make log_dir directory to store checkpoints
        log_dir = self.FILEIO['LOG_DIR']
        logging.info('Make log dir: {}'.format(log_dir))
        utils.makedir(log_dir)
        # Make yaml config file
        yaml_filepath = self.FILEIO['CONFIG_YAML']
        self.make_yaml_file(yaml_filepath)
        
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

class AttrDict(dict):
    IMMUTABLE = '__immutable__'
        
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

def make_cfg():
    cfg = edict()
    cfg.a = 'a'
    cfg.b = 'b'

cfg = make_cfg()

#     __C = AttrDict()
# # Consumers can get config by:
# #   from detectron.core.config import cfg
#     cfg = __C

#     # Random note: avoid using '.ON' as a config key since yaml converts it to True;
#     # prefer 'ENABLED' instead

#     # ---------------------------------------------------------------------------- #
#     # Training options
#     # ---------------------------------------------------------------------------- #
#     __C.TRAIN = AttrDict()

#     # Initialize network with weights from this .pkl file
#     __C.TRAIN.WEIGHTS = ''

#     # Datasets to train on
#     # Available dataset list: detectron.datasets.dataset_catalog.datasets()
#     # If multiple datasets are listed, the model is trained on their union
#     __C.TRAIN.DATASETS = ()

#     # Scales to use during training
#     # Each scale is the pixel size of an image's shortest side
#     # If multiple scales are listed, then one is selected uniformly at random for
#     # each training image (i.e., scale jitter data augmentation)
#     __C.TRAIN.SCALES = (600, )

#     # Max pixel size of the longest side of a scaled input image
#     __C.TRAIN.MAX_SIZE = 1000
#     return cfg
# =================================================================================================================
# MAIN
def main(**kwargs):
    cfg = make_cfg()
    make_yaml_file(cfg, current_dir / 'cfg_test.yaml')
        # # Load input arguments
        # run_opt = get_varargin(kwargs, 'run_opt', 'makefile')
        # default_yaml_filepath = current_dir / 'base_config.yaml'
        # yaml_filepath = get_varargin(kwargs, 'to_file', default_yaml_filepath)
        
        # config_obj = Config()
        # # # config_obj. MODEL.NAME = 'resnet50'
        # config_obj.MODEL.NAME = 'resnet50'
        # if run_opt == 'makefile':
        #         # Make yaml file
        #         config_obj.make_yaml_file(yaml_filepath)
        # else:
        #         temp = config_obj.load_yaml_file(yaml_filepath)
        #         config = edict(temp)                
        #         for key, val in config.items():
        #                 print(key, val)
        # config_obj = Config()
        # config_obj. MODEL.NAME = 'resnet50'
        # yaml_filepath = script_dir / 'base_config.yaml'
        # Make yaml file
        # config_obj.make_yaml_file(yaml_filepath)
        # logging.info('Config yaml file is created: {}'.format(yaml_filepath))
        # Load yaml file
        # print(config.MODEL)
        # print(config.MODEL)
# =================================================================================================================
# DEBUG
#%%
if __name__ == '__main__':
    main()
        # yaml_filepath = Path(current_dir) / 'base_config.yaml'
        # main(run_opt = 'loadfile', to_file = yaml_filepath)
