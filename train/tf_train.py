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
import tensorflow.keras as keras # tensorflow 2.0
print('Tensorflow version: {}'.format(tf.__version__))
print('Keras version: {}'.format(keras.__version__))
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.python.keras.engine as KE
import tensorflow.keras.models as KM
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape

from tensorflow.keras.datasets import mnist

#	Visualization Packages
import matplotlib.pyplot as plt
import skimage.io as skio
#	Utilities
from tqdm import tqdm
import time
import psutil
from datetime import datetime
import logging
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
script_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin, ProgressBar
from utils_dir import logger_utils as logger
from model import resnet as resnet_utils
# from datasets import mnist
from config import baseConfig
logger.logging_setup()
# =================================================================================================================
# DEFINES
timenow = datetime.now().strftime('%Y%m%d_%H%M')
cfg = baseConfig.base_config()
cfg.MODEL.NAME = 'mnist_model'
cfg.TRAIN.NB_EPOCHS = 6
cfg.TRAIN.STEPS_PER_EPOCH = 500
cfg.TRAIN.PROFILING_FREQ = 2
# =================================================================================================================
#%%
def download_model_weight(model_url,**kwargs):
    """
    Download pre-trained model weight .h5 from url
    
    Arguments:
        model_url {str} -- link to .h5 file
    """
    model_dir = cfg.FILEIO.PRE_TRAINED_DIR
    model_filename = os.path.split(model_url)[1]
    to_file = get_varargin(kwargs, 'to_file', os.path.join(model_dir, model_filename))
    # Start downloading
    utils.download_url(model_url, to_file)

# download_model_weight(cfg.MODEL.URL_RESNET50)

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
    model_dir = get_varargin(kwargs, 'model_dir', cfg.FILEIO.LOG_DIR)
    # List .h5 files in model_dir directory
    file_list = utils.select_files(model_dir, ext = ['.h5'])
    return sorted(file_list)[-1]

#%%

def load_model(**kwargs):
    pass

def set_log_dir(**kwargs):
    pass

class profiling_Callback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(profiling_Callback, self).__init__()
        self.nb_epochs = get_varargin(kwargs, 'epochs', cfg.TRAIN.NB_EPOCHS)
        self.steps_per_epoch =  get_varargin(kwargs, 'steps_per_epoch', cfg.TRAIN.STEPS_PER_EPOCH)        
        self.profiling_freq =  get_varargin(kwargs, 'profiling_freq', cfg.TRAIN.PROFILING_FREQ) 
        self.progress = ProgressBar(self.nb_epochs, title = 'Epoch', 
                                    symbol = '=', printer = 'logger')
        self.batchbar = ProgressBar(self.steps_per_epoch, title= 'Batch',
                                    printer = 'stdout')
        
    def on_train_batch_end(self, batch, logs=None):
        self.batchbar.current += 1
        self.batchbar()
        if batch == self.steps_per_epoch-1:
            self.batchbar.current = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress.current += 1
        self.progress()
        
        logging.info('loss: {:.4f} -  Accuracy: {:.4f}'\
            ' - val_loss:{:.4f} - val_accuracy:{:.4f}'\
            .format(logs['loss'], logs['accuracy'],
                    logs['val_loss'], logs['val_accuracy'],
                    ))
        if (epoch) % self.profiling_freq == 0:
            logger.logPC_usage()
            logger.logGPU_usage()

def train_init(**kwargs):
    model_name = get_varargin(kwargs, 'model_name', cfg.MODEL.NAME)
    retrain = get_varargin(kwargs, 'retrain', True)
    init_epoch = 0
    if retrain is False: # Train from scratch
        prefix = datetime.now().strftime('%y%m%d_%H%M') 
        ckpts_logdir = os.path.join(cfg.FILEIO.LOG_DIR, '{}-{}-ckpts'\
            .format(prefix, model_name))
        utils.makedir(ckpts_logdir)
        # cfg.TRAIN.LOG_DIR = 
    else:
        last_model = get_last_model()
        ckpts_logdir = os.path.dirname(last_model)
        filename = os.path.splitext(Path(last_model).name)[0]
        init_epoch = int(filename[-3:])
    return ckpts_logdir, init_epoch

@timeit
def train_model(model, **kwargs):
    logger.log_nvidia_smi_info()
    log_dir = get_varargin(kwargs, 'log_dir', cfg.FILEIO.LOG_DIR)
    # model_name = cfg.MODEL.NAME.lower()
    ckpts_filename = 'mnist_model_ckpts-{epoch:03d}.h5'
    
    ckpts_logdir, init_epoch = train_init(retrain = False)
    ckpts_filepath = os.path.join(ckpts_logdir, ckpts_filename)
    # default_checkpoint = os.path.join(log_dir, '{}_ckpts.h5'.format(model_name))
    # checkpoint_path = get_varargin(kwargs, 'checkpoint', default_checkpoint)
    # Callbacks
    # tfboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir,
    #                                 histogram_freq=0, 
    #                                 write_graph=True, 
    #                                 write_images=False)
    # checkpoint_filename = 'mnist_model_ckpts-{epoch:03d}.h5'
    # checkpoint_path = os.path.join(log_dir, checkpoint_filename)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(ckpts_filepath, 
                                        verbose = 1, 
                                        monitor = 'loss',
                                        save_best_only = True,
                                        save_weights_only=True,
                                        mode = 'min')
    
    logging_callback = profiling_Callback(profiling_freq = cfg.TRAIN.NB_EPOCHS-1)
    # callbacks = [tfboard_callback, checkpoint_callback]
    # Parallel model
    # tf.compat.v1.disable_eager_execution()
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
        # model = model
    # with tf.device("/device:CPU:0"):
    #     model = model
    model = tf.compat.v1.keras.utils.multi_gpu_model(model, gpus = cfg.TRAIN.NB_GPUS,
                                        cpu_merge = True)

    # keras.utils.multi_gpu_model()
    model.compile(loss = keras.losses.categorical_crossentropy,
            optimizer = keras.optimizers.Adadelta(),
            metrics= ['accuracy'])
        
    # callbacks = [checkpoint_callback]
    callbacks = [checkpoint_callback, logging_callback]
    # X_train, y_train, X_test, y_test = mnist.get_mnist()
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.fit(X_train,y_train,
            steps_per_epoch = cfg.TRAIN.STEPS_PER_EPOCH,
            initial_epoch = init_epoch,
            epochs = cfg.TRAIN.NB_EPOCHS,
            verbose = 0,
            validation_data = (X_test, y_test),
            callbacks = callbacks,
            )
# =================================================================================================================
# MAIN
def main(**kwargs):
    # model = mnist_model(num_classes=10)
    num_classes = 10
    # input_shape = (28,28,1)
    # model = KM.Sequential()
    # model.add(KL.Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    # model.add(KL.Conv2D(64, (3, 3), activation='relu'))
    # model.add(KL.MaxPooling2D(pool_size=(2, 2)))
    # model.add(KL.Dropout(0.25))
    # model.add(KL.Flatten())
    # model.add(KL.Dense(128, activation='relu'))
    # model.add(KL.Dropout(0.5))
    # model.add(KL.Dense(num_classes, activation='softmax'))
    # model.summary()
    with tf.device("/device:CPU:0"):
        model = mnist_model(num_classes = num_classes)
    train_model(model)
    # Load mnist data
# main()
# =================================================================================================================
# DEBUG
#%%
if __name__ == '__main__':
    main()