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
# RESNET RELATED
def identity_block(x, **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    kernel_size = get_varargin(kwargs, 'kernel_size', 3)
    strides = get_varargin(kwargs, 'strides', (1,1))
    stage = get_varargin(kwargs, 'stage', 2)
    conv_block = get_varargin(kwargs, 'conv_block', False)
    block_id = get_varargin(kwargs, 'block_id', 1)
    
    if stage == 2:
        filters = config.MODEL['RESNET']['FILTERS_C2']
    elif stage == 3:
        filters = config.MODEL['RESNET']['FILTERS_C3']
    elif stage == 4:
        filters = config.MODEL['RESNET']['FILTERS_C4']
    else:
        filters = config.MODEL['RESNET']['FILTERS_C5']
    filters1, filters2, filters3 = filters
    bn_axis = 3 # Channel last, tensorflow backend
    prefix_blockname = 'C{}_branch2_blk{}_'.format(stage, block_id)    
    fx = KL.Conv2D(filters1, (1, 1), strides = strides, name = '{}Conv1'.format(prefix_blockname))(x)
    fx = KL.BatchNormalization(axis=bn_axis, name = '{}Bnorm1'.format(prefix_blockname))(fx) #bn: batchnorm
    fx = KL.Activation('relu', name = '{}Act1'.format(prefix_blockname))(fx)

    fx = KL.Conv2D(filters2, kernel_size,padding='same', name = '{}Conv2'.format(prefix_blockname))(fx)
    fx = KL.BatchNormalization(axis = bn_axis, name = '{}Bnorm2'.format(prefix_blockname))(fx)
    fx = KL.Activation('relu', name = '{}Act2'.format(prefix_blockname))(fx)

    fx = KL.Conv2D(filters3, (1, 1), name = '{}Conv3'.format(prefix_blockname))(fx)
    fx = KL.BatchNormalization(axis=bn_axis, name = '{}Bnorm3'.format(prefix_blockname))(fx)
    
    prefix_blockname = 'C{}_branch1_blk{}_'.format(stage, block_id)    
    #Shortcut branch     
    if conv_block is True:
        shortcut = KL.Conv2D(filters3, (1, 1), strides = strides, name = '{}Conv1'.format(prefix_blockname))(x)
        shortcut = KL.BatchNormalization(axis=bn_axis, name = '{}Bnorm1'.format(prefix_blockname))(shortcut)
    else:
        shortcut = x
        
#   Merge
    fx = KL.Add()([fx, shortcut])
    fx = KL.Activation('relu')(fx)
    return fx
# =================================================================================================================
# RESNET50
def ResNet50(include_top=True,
             input_tensor=None,
             pooling=None,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    # Arguments       
    # Returns
        A Keras model instance.
    """
    # Input arguments
    include_top = get_varargin(kwargs, 'include_top', True)
    nb_classes = get_varargin(kwargs, 'nb_classes', 1000)
    default_input_shape = _obtain_input_shape(None,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)
    input_shape = get_varargin(kwargs, 'input_shape', default_input_shape)
    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1        

    x = KL.ZeroPadding2D((3, 3))(img_input)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = KL.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    for stage, nb_block in zip([2,3,4,5], [3,4,6,3]):
        for blk in range(nb_block):
            conv_block = True if blk == 0 else False
            strides = (2,2) if stage>2 and blk==0 else (1,1)           
            x = identity_block(x, stage = stage, block_id = blk + 1,
                               conv_block = conv_block, strides = strides)
            
    x = KL.AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = KL.Flatten()(x)
        x = KL.Dense(nb_classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = KL.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = KL.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x, name='resnet50')
    return model
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    # model = ResNet50(include_top=True)
    download_model_weight(URL_RESNET50_NOTOP)
    # model.summary()
    # export_keras_model_structure(model, 'keras_model_structure.json')