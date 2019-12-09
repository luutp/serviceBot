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
#%%
from __future__ import print_function
import os
import inspect, sys
import argparse
#	File IO
import json
from pathlib import Path
import h5py
from contextlib import redirect_stdout

#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#   AI Framework
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
import tensorflow_datasets as tfds

# tf.compat.v1.disable_eager_execution()

import tensorflow.keras as keras # tensorflow 2.0
print('Tensorflow version: {}'.format(tf.__version__))
print('Keras version: {}'.format(keras.__version__))
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.python.keras.engine as KE
import tensorflow.keras.models as KM
from tensorflow.keras.models import Model

from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.core import Lambda
# from tensorflow.keras.models import Model
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
import config.baseConfig as baseConfig

logger.logging_setup()
# =================================================================================================================
# DEFINES
timenow = datetime.now().strftime('%Y%m%d_%H%M')
cfg = baseConfig.base_config()
cfg.MODEL.NAME = 'mnist_model'
cfg.TRAIN.NB_EPOCHS = 2
cfg.TRAIN.STEPS_PER_EPOCH = 1000
cfg.TRAIN.PROFILING_FREQ = cfg.TRAIN.NB_EPOCHS - 1
cfg.TRAIN.MAX_CKPTS_FILES = 3
# =================================================================================================================
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

    model = KM.Model(inputs = img_input, outputs = x)
    return model

def get_epoch_number(filename):
    filename = Path(filename).name
    filename = filename.split(os.extsep)[0]
    epoch = int(filename[filename.rfind('-')+1:])
    return epoch

def get_last_model(**kwargs):
    log_dir = get_varargin(kwargs, 'log_dir', cfg.FILEIO.LOG_DIR)
    and_key = get_varargin(kwargs, 'and_key', ['.ckpt'])
    or_key = get_varargin(kwargs, 'or_key', None)
    # List .h5 files in model_dir directory
    file_list = utils.select_files(log_dir, and_key = and_key, or_key = or_key )
    return sorted(file_list)[-1]

def load_model(**kwargs):
    pass


class profiling_Callback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(profiling_Callback, self).__init__()
        self.nb_epochs = get_varargin(kwargs, 'epochs', cfg.TRAIN.NB_EPOCHS)
        self.steps_per_epoch =  get_varargin(kwargs, 'steps_per_epoch', cfg.TRAIN.STEPS_PER_EPOCH)        
        self.profiling_freq =  get_varargin(kwargs, 'profiling_freq', cfg.TRAIN.PROFILING_FREQ) 
        
    def on_train_begin(self, logs = None):
        self.end_epoch = self.params['epochs']
        self.start_epoch = self.end_epoch - self.nb_epochs
        
        self.progress = ProgressBar(self.end_epoch,
                                    title = 'Epoch', 
                                    start = self.start_epoch,
                                    symbol = '=', printer = 'logger')
        logging.info('Train params:')
        logging.info(json.dumps(self.params, indent = 2))
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress.current = epoch+1
        self.progress()
        logging.info('loss: {:.4f} - accuracy: {:.4f}'\
            ' - val_loss: {:.4f} - val_accuracy: {:.4f}'\
            .format(logs['loss'], logs['accuracy'],
                    logs['val_loss'], logs['val_accuracy']))
        if epoch+1 == self.params['epochs']:
            logger.logPC_usage()
            logger.log_nvidia_smi_info()
        # # Delete ckpts file, only keep n-latest files
        last_model = get_last_model()
        ckpts_dir = Path(last_model).parent
        ckpts_filelist = sorted(utils.select_files(ckpts_dir,and_key = ['.ckpt'])
                                        , reverse = True)
        epoch_list = sorted(np.unique([get_epoch_number(e) for e in ckpts_filelist]),
                            reverse = True)
        if len(epoch_list) > cfg.TRAIN.MAX_CKPTS_FILES:
            cut_epoch = epoch_list[cfg.TRAIN.MAX_CKPTS_FILES-1]
            for f in ckpts_filelist:
                if get_epoch_number(f) < cut_epoch:
                    os.remove(f)
        
def train_init(model, **kwargs):
    model_name = get_varargin(kwargs, 'model_name', cfg.MODEL.NAME)
    retrain = get_varargin(kwargs, 'retrain', True)
    logging.info('Retrain Model: {}'.format(retrain))
    init_epoch = 0
    if retrain is False: # Train from scratch
        prefix = datetime.now().strftime('%y%m%d_%H%M') 
        ckpts_logdir = os.path.join(cfg.FILEIO.LOG_DIR, '{}-{}-ckpts'\
            .format(prefix, model_name))
        utils.makedir(ckpts_logdir)
        model = model
    else:
        last_model = get_last_model(log_dir = cfg.FILEIO.LOG_DIR)
        ckpts_logdir = os.path.dirname(last_model)
        last_model = last_model.replace('.index','')
        filename = os.path.splitext(Path(last_model).name)[0]
        init_epoch = int(filename[filename.rfind('-')+1:])
        logging.info('Load model weights from: {}'.format(Path(last_model).name))
        model.load_weights(last_model)
        # model = KM.load_model(last_model)
    # Save cfg to yaml file
    yaml_filepath = Path(ckpts_logdir) / '{}-config.yaml'.format(cfg.MODEL.NAME.lower())
    baseConfig.make_yaml_file(cfg, yaml_filepath)    
    return model, ckpts_logdir, init_epoch

@timeit
def train_model(**kwargs):
    ckpts_filename = 'mnist_model-ckpts-{epoch:04d}.ckpt'
    hist_filename = '{}-hist.csv'.format(cfg.MODEL.NAME.lower())
    retrain_opt = True
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

    # mnist_train, mnist_test = datasets['train'], datasets['test']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows = 28
    img_cols = 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    
    mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    strategy = tf.distribute.MirroredStrategy() 
    BUFFER_SIZE = 1000
    BATCH_SIZE_PER_REPLICA = 32
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)   
   
    with strategy.scope():
        model = mnist_model(10)
        model, ckpts_logdir, start_epoch = train_init(model, retrain = retrain_opt)
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer = keras.optimizers.Adam(),
                metrics=['accuracy'])
    
    ckpts_filepath = os.path.join(ckpts_logdir, ckpts_filename)
    hist_filepath = os.path.join(ckpts_logdir, hist_filename)
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = ckpts_filepath,
                                                          verbose = 1,
                                                          save_weights_only = True)
    logging_callback = profiling_Callback()
    csvlog_callback = keras.callbacks.CSVLogger(hist_filepath,
                                                append = True)
    
    callbacks = [checkpoint_callback, logging_callback, csvlog_callback]
    model.fit(train_dataset, 
              initial_epoch = start_epoch,
              epochs = start_epoch + cfg.TRAIN.NB_EPOCHS, 
              validation_data = eval_dataset,
              verbose = True, 
              callbacks=callbacks)
    
def load_df(fullfilename, **kwargs):
    """Load .csv or .xlxs file to pandas dataframe
    Input:
        fullfilename: fullpath to input data file in .csv or .xlsx format
        Options:
            skiprows: row index to skip
    Returns:
        df: pandas dataframe
    """
    skiprows = get_varargin(kwargs, 'skiprows',None)
    filename,file_ext = os.path.splitext(fullfilename)
    logging.info('Pandas read: {}{}'.format(filename, file_ext))    
    if file_ext == '.csv':        
        df = pd.read_csv(fullfilename, skiprows = skiprows)
    else:
        df = pd.read_excel(fullfilename, skiprows = skiprows)    
    # ==== END ====
    logging.info('DONE: %s' % inspect.stack()[0][3])     
    return df

def load_csv_history(**kwargs):
    csv_filepath = Path(get_last_model()).parent / '{}-hist.csv'.format(cfg.MODEL.NAME.lower())
    print(csv_filepath)
    df = load_df(csv_filepath)
    history = df.to_dict('list')
    return history
    # return history

def plt_train_history(history): 
    """Generate plot for model loss and accuracy from training history
    Args:
        history ([type]): [description] 
    """
    # save_opt = get_varargin(kwargs, 'save', False)
    # figname = get_varargin(kwargs, 'figname', '{}-model_loss.png'.format(todaystr))
     
    fig = plt.figure(figsize = (10,5))
    # Loss 
    ax = fig.add_subplot(121)
    plt.plot(history['epoch'],history['loss'], color = 'k' ) # Training
    plt.plot(history['epoch'],history['val_loss'], color = 'r') # Validation
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # Accuracy
    ax = fig.add_subplot(122)
    plt.plot(history['epoch'],history['accuracy'], color = 'k')
    plt.plot(history['epoch'],history['val_accuracy'], color = 'r')
    # Annotation
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

# history = load_csv_history()
# plt_train_history(history)

# =================================================================================================================
# MAIN
def main(**kwargs):
    # model = mnist_model(num_classes=10)
    num_classes = 10
    train_model()
    # Load mnist data
# =================================================================================================================
# DEBUG
#%%
if __name__ == '__main__':
    main()