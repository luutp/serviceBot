#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
coco_utils.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/30
'''
# =================================================================================================================
# IMPORT PACKAGES
from __future__ import print_function
import os
import numpy as np
import logging
import json
from pathlib import Path
from tqdm import tqdm
from skimage import measure, io
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
from utils import get_varargin, timeit
import logger_utils
# =================================================================================================================
# SETUP
logger_utils.logging_setup()
#	Working Directories
current_dir = os.getcwd()
user_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
# 	Logging
# =================================================================================================================
class coco_dataset(object):
# Initialize Class
    def __init__(self, **kwargs):
        json_filepath = get_varargin(kwargs, 'json_filepath', 'untitiles.json')
        imgpath = get_varargin(kwargs,'imgpath', os.path.dirname(json_filepath))
        self.json_filepath = json_filepath
        self.imgpath = imgpath
        self.load_json()
# =================================================================================================================
    @property
    def json_filepath(self):
        return self._json_filepath
    @json_filepath.setter
    def json_filepath(self, filepath):
        self._json_filepath = filepath
# =================================================================================================================
# METHODS
    def load_json(self, **kwargs):
        """
        Load .json file and create json object
        """
        if not os.path.isfile(self.json_filepath):
            logging.error('{} is not exist'.format(self.json_filepath))
            raise FileNotFoundError('{} is not exist'.format(self.json_filepath))
        else:
            with open(self.json_filepath) as fid:
                logging.info('Open file: {}'.format(self.json_filepath))
                self.json_obj = json.load(fid)
    
    def get_keys(self, **kwargs):
        key_list = list(self.json_obj.keys())
        display_opt = get_varargin(kwargs, 'display', False)
        self.json_key_list = key_list
        if display_opt is True:
            print(key_list) 
    
    def get_json_data(self, **kwargs):
        """
        Get key value from json object
        Options:
            display: Print key values info. Default: True
            keys: List of key to get value. Default: All key
        """
        display_opt = get_varargin(kwargs, 'display', True)
        self.get_keys()
        input_keys = get_varargin(kwargs, 'keys', self.json_key_list)
        for key in input_keys:
            if key in self.json_key_list:
                strcmd = "self.{} = self.json_obj.get('{}')".format(key,key)
                exec(strcmd)
                if display_opt is True:
                    strcmd = "print(self.{})".format(key)
                    exec(strcmd)
            else:
                logging.info('Key not exist in coco json file: {}'.format(key))
        
    def get_info(self, **kwargs):
        """
        Get values from 'info' key from json file object
        """
        display_opt = get_varargin(kwargs, 'display', True)
        self.get_json_data(keys = ['info'], display = display_opt)
        return self.info
    
    def get_licenses(self, **kwargs):
        """
        Get license info from coco .json file
        """        
        display_opt = get_varargin(kwargs, 'display', True)
        self.get_json_data(keys = ['licenses'], display = display_opt)
        return self.licenses
    
    def get_categories(self, **kwargs):
        """
        Get categories information from .json file
        """
        display_opt = get_varargin(kwargs, 'display', True)
        self.get_json_data(keys = ['categories'], display = display_opt)
        print('Number of Categories: {}'.format(len(self.categories)))
        return self.categories

    def get_images(self, **kwargs):
        """
        Get images information from .json file
        """
        display_opt = get_varargin(kwargs, 'display', False)
        self.get_json_data(keys = ['images'], display = display_opt)
        print('Number of Images: {}'.format(len(self.images)))
        print('Example Image Info:')
        print('\t{}'.format(self.images[0]))
        return self.images
    
    def get_annotations(self, **kwargs):
        """
        Get annotations information from .json file
        """
        display_opt = get_varargin(kwargs, 'display', False)
        self.get_json_data(keys = ['annotations'], display = display_opt)
        print('Number of Annotations: {}'.format(len(self.annotations)))
        print('Example Annotations Info:')
        print('\t{}'.format(self.annotations[0]))
        return self.annotations

# =================================================================================================================
# STATIC METHODS
    @staticmethod
    def function1(**kwargs):
        pass

def main():
    json_obj = coco_json()
    json_obj.json_filepath = os.path.join(os.path.dirname(script_dir),
                                        'Downloads\\coco_annotations.json')
    json_obj.load_json()
    json_obj.get_info()
# DEBUG
if __name__ == '__main__':
    main()