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
#%%
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
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin
from utils_dir import logger_utils as logger
from model import resnet as resnet_utils
from datasets import mnist
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
#%%
def download_model_weight(model_url,**kwargs):
    """
    Download pre-trained model weight .h5 from url
    
    Arguments:
        model_url {str} -- link to .h5 file
    """
    model_dir = PRE_TRAINED_DIR
    model_filename = os.path.split(model_url)[1]
    to_file = get_varargin(kwargs, 'to_file', os.path.join(model_dir, model_filename))
    # Start downloading
    utils.download_url(model_url, to_file)

def get_nb_gpus():
    return len(K._get_available_gpus())

def compile_model(**kwargs):
    pass

def mnist_model(num_classes, **kwargs):
    # Reset default graph. Keras leaves old ops in the graph,
    # which are ignored for execution but clutter graph
    # visualization in TensorBoard.
    # tf.reset_default_graph()
    input_shape = (28,28,1)
    img_input = KL.Input(shape = input_shape, name="input_image")
    x = KL.Conv2D(32, (3, 3), padding="same",name="conv1")(img_input)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(64, (3, 3), padding="same",name="conv2")(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = KL.Flatten(name="flat1")(x)
    x = KL.Dense(128, activation='relu', name="dense1")(x)
    x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

    model = Model(inputs = img_input, outputs = x)
    return model


def parallel_model(model, **kwargs):
    pass

def get_last_model(**kwargs):
    model_dir = get_varargin(kwargs, 'model_dir', config.FILEIO['LOG_DIR'])
    # List .h5 files in model_dir directory
    utils.select_files(model_dir, ext = '.h5')

def load_model(**kwargs):
    pass

def set_log_dir(**kwargs):
    pass

def train_model(model, **kwargs):
    logger.log_nvidia_smi_info()
    log_dir = get_varargin(kwargs, 'log_dir', config.FILEIO['LOG_DIR'])
    model_name = config.MODEL['NAME'].lower()
    default_checkpoint = Path(log_dir) / '{}_ckpts.h5'.format(model_name)
    checkpoint_path = get_varargin(kwargs, 'checkpoint', default_checkpoint)
    # Callbacks
    tfboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir,
                                    histogram_freq=0, 
                                    write_graph=True, 
                                    write_images=False)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,                                        
                                        verbose = 1, 
                                        monitor = 'loss',
                                        save_best_only = True,
                                        save_weights_only=True,
                                        mode = 'min')
    callbacks = [tfboard_callback, checkpoint_callback]
# =================================================================================================================
# MAIN
def main(**kwargs):
    
    model = mnist_model(num_classes=10)
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=[tensorboard_callback])
    # Load mnist data
    X_train, y_train, X_test, y_test = mnist.get_mnist()
    
# =================================================================================================================
# DEBUG
#%%
if __name__ == '__main__':
    main()