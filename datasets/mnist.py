#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
mnist.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/11/13
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
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#   AI framework
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.datasets import mnist
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
from utils_dir import logger_utils as logger
from datasets import dataset_utils
logger.logging_setup()
# =================================================================================================================
# GET DATASETS
def get_mnist(**kwargs):
        # input image dimensions
    img_rows, img_cols = 28, 28
    nb_classes = 10
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    return X_train, y_train, X_test, y_test

def plt_samples(X_train, y_train):
    fig = plt.figure(figsize = (6,6))
    nb_rows = 3
    nb_cols = 3
    for i in range(nb_rows * nb_cols):
        idx = random.choice(range(100))
        image = X_train[idx,:,:,0].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
        label = np.argmax(y_train[idx,:])
        ax = fig.add_subplot(nb_rows, nb_cols, i+1)          # subplot with size (width 3, height 5)
        plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.        
        # plt.title('{}'.format(int(y_train[idx]))
        plt.axis('off')  # do not show axis value
        plt.tight_layout()   # automatic padding between subplots    
        ax.set_title('label:{}'.format(label))

# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    pass