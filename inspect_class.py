#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pyqt.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/01/15
'''
# =================================================================================================================
# IMPORT PACKAGES
#%%
from __future__ import print_function
from IPython.core.debugger import set_trace

import os
import inspect, sys
import argparse
#	File IO
import json
import functools
#	Data Analytics
import pandas as pd
import numpy as np
import random
from easydict import EasyDict as edict
#  PyQT
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#	Visualization Packages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import matplotlib.widgets as mpl_widgets
from  matplotlib.path import Path
import skimage.io as skio
# Custom packages
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
project_dir = os.path.join(user_dir, 'serviceBot')
if not (project_dir in sys.path):
    print('sys.append: {}'.format(project_dir))
    sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin, verbose
import  utils_dir.utils as utils
from utils_dir import logger_utils as logger_utils
from icons.iconClass import iconClass
pyIcons = iconClass(icon_dir = os.path.join(project_dir, 'icons'))
logger = logger_utils.htmlLogger(log_file = os.path.join(project_dir, '{}_GUIlogging.html'\
    .format(datetime.now().strftime('%y%m%d'))), mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))

# =================================================================================================================
# BEGIN
#%%
class classExplorer(object):
    def __init__(self, **kwargs):
        class_name = get_varargin(kwargs, 'class_name', None)
        self._class_name = class_name
    
    @property
    def class_name(self):
        return self._class_name
    @class_name.setter
    def class_name(self, inputVal):
        self._class_name = inputVal
    
    @property
    def members(self):
        members = inspect.getmembers(self.class_name)
        return members
    
    @property
    def attr_members(self):
        attr_members = []
        for mem in self.members:
            if not inspect.isfunction(mem[1]) and not inspect.ismethod(mem[1]):
                attr_members.append(mem)
        return attr_members
    
    @property
    def function_members(self):
        func_members = []
        for mem in self.members:
            if inspect.isfunction(mem[1]):
                func_members.append(mem)
        return func_members

    def list_attrs(self, **kwargs):
        key = get_varargin(kwargs, 'key', 'all')
        attrs = []
        for mem in self.attr_members:
            if not mem[0].startswith('_'):
                attrs.append(mem[0])
        return self.match_key(attrs, key)
    
    def list_function_names(self, **kwargs):
        key = get_varargin(kwargs, 'key', 'all')
        func_names = []
        for mem in self.function_members:
            if not mem[0].startswith('_'):
                func_names.append(mem[0])
        return self.match_key(func_names, key)
        
    def list_private_function_names(self, **kwargs):
        key = get_varargin(kwargs, 'key', 'all')
        func_names = []
        for mem in self.function_members:
            if mem[0].startswith('_'):
                func_names.append(mem[0])
        return self.match_key(func_names, key)
    
    def show_docstring(self, input_name):
        for mem in self.function_members:
            if input_name == mem[0]:
                print(inspect.getsource(mem[1]))
    # STATIC METHODS
    @staticmethod
    def match_key(input_list, key):
        if key == 'all':
            return input_list
        else:
            new_list = [item for item in input_list if key in item]
            return new_list

def main():
    exp = classExplorer(class_name = PyQt5.PyWidgets)
    # display(exp.methods)
    display(exp.list_function_names(key = 'all'))
    # display(exp.function_members)
    # exp.show_docstring('get_verts')
    # instance = eval('PyQt5.QtWidgets')
    # print(type(instance))
    # clsmembers = inspect.getmembers(instance, inspect.isclass)
    # display(clsmembers)
    # display(inspect.getsource(exp.function_members[30][1]))
    # print(inspect.getsource(mpl_widgets.Circle.get_path))
    
    
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()
