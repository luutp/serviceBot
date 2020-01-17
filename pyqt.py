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
import os
import inspect, sys
import argparse
#	File IO
import json
from pathlib import Path
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
import matplotlib.pyplot as plt
import skimage.io as skio
# Custom packages
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    
import cv2
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
icons_dir = os.path.join(project_dir, 'icons')
logger = logger_utils.htmlLogger(log_file = os.path.join(project_dir, '{}_GUIlogging.html'\
    .format(datetime.now().strftime('%y%m%d'))), mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))
# =================================================================================================================
# FUNCTIONS
# Decorator
def log_info(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        logger.info('START: {}'.format(func.__name__))
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(e)
            return None
        msg = 'DONE: {func_name}.\t' \
            .format(func_name = func.__name__)        
        logger.info(msg)
        return result
    return inner
#%%
def show_msg_dialog(**kwargs):
    msg_type = get_varargin(kwargs, 'type', 'info')
    title = get_varargin(kwargs, 'title', 'Message Dialog')
    message = get_varargin(kwargs, 'message', 'Custom message')
    
    msg = QMessageBox()
    if msg_type == 'info':
        msg.setIcon(QMessageBox.Information)    
    elif msg_type == 'warning':
        msg.setIcon(QMessageBox.Warning)    
    elif msg_type == 'question':
        msg.setIcon(QMessageBox.Question)    
    msg.setText(message)
    msg.setWindowTitle(title)
    msg.exec_()
    
def uigridcomp(**kwargs):
    layout_type = get_varargin(kwargs, 'layout', 'horizontal') 
    comp_list = get_varargin(kwargs, 'components', None)
    nb_comps = len(comp_list)
    icon_path_list = get_varargin(kwargs, 'icons', [None]*nb_comps)
    label_list = get_varargin(kwargs, 'labels', [None]*nb_comps)
    ratio_list = get_varargin(kwargs, 'ratios', [np.floor(100/nb_comps)]*nb_comps)
    
    if layout_type == 'vertical':
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    
    def create_slider(**kwargs):
        slider_type = get_varargin(kwargs, 'type', 'horizontal')
        if slider_type == 'horizontal':
            py_comp = QSlider(Qt.Horizontal)
        else:
            py_comp = QSlider(Qt.Vertical)
        py_comp.setMinimum(0)
        py_comp.setMaximum(100)
        py_comp.setValue(50)
        py_comp.setTickInterval(10)
        py_comp.setTickPosition(QSlider.TicksBelow)
        return py_comp
    
    output_widgets = []
    for comp, label, icon, ratio in zip(comp_list, label_list, icon_path_list, ratio_list):
        py_comp = None
        if comp == 'pushbutton':
            py_comp = QPushButton(label)
            py_comp.setIcon(QIcon(QPixmap(icon)))
        elif comp == 'togglebutton':
            py_comp = QPushButton(label)
            py_comp.setIcon(QIcon(QPixmap(icon)))
            py_comp.setCheckable(Tru)
            py_comp.toggle()
        elif comp == 'hSlider':
            py_comp = create_slider(type = 'horizontal')
        elif comp == 'vSlider':
            py_comp = create_slider(type = 'vertical')
        elif comp == 'edit':
            py_comp = QLineEdit(label)
        elif comp == 'label':
            py_comp = QLabel(label)
        elif comp == 'list':
            py_comp = QListWidget()
            for l in label:
                py_comp.addItem(l)
        elif comp == 'blank':
            layout.addStretch(0)
        if comp != 'blank':
            output_widgets.append(py_comp)
            layout.addWidget(py_comp, ratio)
    return layout, output_widgets

class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.icons_dir = icons_dir
        self.layout1, self.layout1_widgets = uigridcomp(components = ['pushbutton', 'edit'], 
                                           icons = [os.path.join(self.icons_dir, 'python.png'), None, None],
                                           labels = ['Browse...', ''],
                                           ratios = [20, 80])  
        self.layout2, self.layout2_widgets = uigridcomp(components = ['list'], 
                                           icons = [None],
                                           labels = [['List Item 1', 'List Item 2', 'List Item 3']])  
        self.layout3, self.layout3_widgets = uigridcomp(layout = 'horizontal', components = ['hSlider', 'edit'],
                                                        ratios = [75, 25])  
        print(dir(self.layout3_widgets[0]))
        self.set_layout()
        self.define_callback()
        
    def set_layout(self):
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.layout1)
        self.layout.addLayout(self.layout2)
        self.layout.addLayout(self.layout3)
        self.setLayout(self.layout)        
    
    def initialize(self):
        pass
    
    # Define callback functions
    def define_callback(self):
        self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        self.layout3_widgets[0].valueChanged.connect(lambda:self.slider_edit_callback(self.layout3_widgets[0]))
        self.layout3_widgets[1].editingFinished.connect(lambda:self.edit_slider_callback(self.layout3_widgets[1]))
        
    # Callbacks
    @log_info
    def pushbutton_browse_callback(self, hObject):
        print('button pressed')
        self.layout1_widgets[1].setText('Browsing...')
        
    @log_info
    def slider_edit_callback(self, hObject):
        edit = self.layout3_widgets[1]
        val = hObject.value()
        edit.setText('{:d}'.format(val))
        
    @log_info
    def edit_slider_callback(self, hObject):
        slider = self.layout3_widgets[0]
        str_val = hObject.text()
        val = float(str_val)
        if val > slider.maximum():
            show_msg_dialog(type = 'warning', message = 'Input > slider.maximum')
            val = slider.maximum()
        elif val < slider.minimum():
            show_msg_dialog(type = 'warning', message = 'Input < slider.minimum')
            val = slider.minimum()
        slider.setValue(val)
            
def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
    window = Form()
    window.setGeometry(4000, 100, 500, 500)
    window.setWindowTitle("PyQT Luu")
    window.show()
    app.exec_()
main()
#%%
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()

# %%
