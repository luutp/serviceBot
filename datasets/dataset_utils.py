#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
utils_dataset.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/28
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
# Data analytics packages
import numpy as np
from google_images_download import google_images_download
# AI framework
import tensorflow as tf
import tensorflow.keras as keras
#  Visualization packages
import matplotlib.pyplot as plt
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
# Custom packages
user_dir = os.path.expanduser('~')
script_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin
from utils_dir import logger_utils as logger
import datasets.dataset_utils
logger.logging_setup()
# =================================================================================================================
# FUNCTIONS
# Display datasets information
def show_traintest_info(X_train, y_train, X_test, y_test, **kwargs):    
    """[summary]
    
    Arguments:
        X_train {[type]} -- [description]
        y_train {[type]} -- [description]
        X_test {[type]} -- [description]
        y_test {[type]} -- [description]
    """
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))
#     Plot label distribution
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(121)
    n, bins, patches = plt.hist(x=np.argmax(y_train, axis = 1), bins=30,
                                alpha=0.7, rwidth=1)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Train Set')
    
    ax = fig.add_subplot(122)
    n, bins, patches = plt.hist(x=np.argmax(y_test, axis = 1), bins=30,
                                alpha=0.7, rwidth=1)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Test Set')
    plt.show()
# =================================================================================================================
# Plot training samples
def plt_samples(img_list, labels, **kwargs):
    figsize = get_varargin(kwargs, 'figsize', (6,6))
    nb_cols = get_varargin(kwargs, 'nb_cols', 3)
    nb_img = len(img_list)
    nb_rows = np.ceil(nb_img/nb_cols)
    fig = plt.figure(figsize = figsize)
    for idx, (img, label) in enumerate(zip(img_list, labels)):
        ax = fig.add_subplot(nb_rows, nb_cols, idx+1)
        plt.imshow(img)
        ax.set_title(label)
# =================================================================================================================

# Download images from google search
@timeit
def downloadimages(query, **kwargs): 
    # keywords is the search query 
    # format is the image file format 
    # limit is the number of images to be downloaded 
    # print urs is to print the image file url 
    # size is the image size which can 
    # be specified manually ("large, medium, icon") 
    # aspect ratio denotes the height width ratio 
    # of images to download. ("tall, square, wide, panoramic") 
    output_dir = get_varargin(kwargs, 'output_directory', os.getcwd())
    nb_images = get_varargin(kwargs, 'nb_images', 4)
    img_size = get_varargin(kwargs, 'size', 'medium')
    
    arguments = {"keywords" : query, 
                 "format" : "jpg", 
                 "limit": nb_images, 
                 "print_urls" : True,
                 'output_directory' : output_dir,
                 "size" : img_size} 
    # creating object 
    response = google_images_download.googleimagesdownload()  
    try: 
        response.download(arguments) 
    # Handling File NotFound Error     
    except FileNotFoundError:  
        pass
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    # output_dir = current_dir
    # downloadimages('cat', nb_images = 5, output_directory = output_dir)
    (X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()
    traintest_info(X_train, y_train, X_test, y_test)
