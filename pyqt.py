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
    font = get_varargin(kwargs, 'font', QFont('Arial', 11, QFont.Normal))
    groupbox = get_varargin(kwargs, 'groupbox', None)
    groupbox_icon = get_varargin(kwargs, 'groupbox_icon', None)
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
            py_comp.setIcon(QIcon(icon))
        elif comp == 'togglebutton':
            py_comp = QPushButton(label)
            py_comp.setIcon(QIcon(icon))
            py_comp.setCheckable(Tru)
            py_comp.toggle()
        elif comp == 'hSlider':
            py_comp = create_slider(type = 'horizontal')
        elif comp == 'vSlider':
            py_comp = create_slider(type = 'vertical')
        elif comp == 'edit':
            py_comp = QLineEdit(label)
        elif comp == 'combobox':
            py_comp = QComboBox()            
            py_comp.setInsertPolicy(QComboBox.InsertAtTop)
            py_comp.addItems(label)
        elif comp == 'label':
            py_comp = QLabel(label)
        elif comp == 'list':
            py_comp = QListWidget()
            py_comp.addItems(label)
            # py_comp.setSelectionMode(QAbstractItemView.MultiSelection)
        elif comp == 'radio':
            py_comp = QRadioButton(label)
            py_comp.setIcon(QIcon(icon))
        elif comp == 'blank':
            layout.addStretch(0)
        if comp != 'blank':
            py_comp.setFont(font)
            output_widgets.append(py_comp)
            layout.addWidget(py_comp, ratio)
        
    if groupbox is not None:
        groupbox_obj = QGroupBox(groupbox)
        gbox_font = font
        gbox_font.setBold(True)
        groupbox_obj.setFont(gbox_font)
        groupbox_obj.setLayout(layout)
        return groupbox_obj, output_widgets
    else:
        return layout, output_widgets

class Form(QWidget):
    def __init__(self):
        super().__init__()
        self.layout1, self.layout1_widgets = uigridcomp(components = ['pushbutton', 'combobox'], 
                                           icons = [pyIcons.fileIO.python, None, None],
                                           labels = ['Browse...', ['/home/phatluu', project_dir]],
                                           groupbox = 'Explorer', 
                                           ratios = [20, 80])  
        self.layout2, self.layout2_widgets = uigridcomp(components = ['list'], 
                                           icons = [None],
                                           labels = [['List Item 1', 'List Item 2', 'List Item 3']])  
        # self.layout2, self.layout2_widgets = uigridcomp(components = ['radio','radio','radio'], 
        #                                    icons = [pyIcons.fileIO.folder, None, None],
        #                                    labels = ['File 1', 'File 2', 'File 3'],
        #                                    groupbox = 'Radio Group')  
        
        self.layout3, self.layout3_widgets = uigridcomp(layout = 'horizontal', components = ['hSlider', 'edit'],
                                                        ratios = [75, 25],
                                                        groupbox = 'Slider') 
        self.set_layout()
        self.define_callback()
        
    def set_layout(self):
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.layout1)
        self.layout.addLayout(self.layout2)
        self.layout.addWidget(self.layout3)
        self.setLayout(self.layout)        
    
    def initialize(self):
        pass
    
    # Define callback functions
    def define_callback(self):
        self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        self.layout1_widgets[1].currentIndexChanged.connect(lambda:self.combobox_browse_callback(self.layout1_widgets[1]))
        self.layout2_widgets[0].doubleClicked.connect(lambda:self.listbox_browse_callback(self.layout2_widgets[0]))
        self.layout3_widgets[0].valueChanged.connect(lambda:self.slider_edit_callback(self.layout3_widgets[0]))
        self.layout3_widgets[1].editingFinished.connect(lambda:self.edit_slider_callback(self.layout3_widgets[1]))
        
    # Callbacks
    @log_info
    def pushbutton_browse_callback(self, hObject):
        dlg = QFileDialog(self, 'Select Directory', project_dir)
        dlg.setFileMode(QFileDialog.Directory)
        sel_dir = None
        if dlg.exec_():
            sel_dir = dlg.selectedFiles()[0]
        if sel_dir is not None:            
            self.layout1_widgets[1].insertItem(0, QIcon(pyIcons.fileIO.folder), sel_dir)
            self.layout1_widgets[1].setCurrentIndex(0)
    
    @log_info
    def combobox_browse_callback(self, hObject):
        currentVal = hObject.currentText()
        file_list = utils.select_files(currentVal, depth = 'root')
        self.layout2_widgets[0].clear()
        for f in file_list:
            item = QListWidgetItem(f)
            file_type = os.path.splitext(f)[1]
            if not file_type:
                item.setIcon(QIcon(pyIcons.fileIO.folder))                
            elif file_type == '.py':
                item.setIcon(QIcon(pyIcons.fileIO.python))
            elif file_type == '.html':
                item.setIcon(QIcon(pyIcons.fileIO.html))
            elif file_type == '.ipynb':
                item.setIcon(QIcon(pyIcons.fileIO.ipynb))
            elif file_type == '.yaml':
                item.setIcon(QIcon(pyIcons.fileIO.yaml))
            elif file_type == '.json':
                item.setIcon(QIcon(pyIcons.fileIO.json))
            elif set([file_type]).issubset(set(['.png', '.jpg', '.svg','.gif'])):
                item.setIcon(QIcon(pyIcons.fileIO.image))
            elif set([file_type]).issubset(set(['.xlsx', '.csv', '.xls'])):
                item.setIcon(QIcon(pyIcons.fileIO.excel))
            elif set([file_type]).issubset(set(['.txt', '.md'])):
                item.setIcon(QIcon(pyIcons.fileIO.txt))
            else:
                pass 
            self.layout2_widgets[0].addItem(item)
        self.layout2_widgets[0].setIconSize(QSize(16,16)) 
        
    @log_info
    def listbox_browse_callback(self, hObject):
        sel_item = [str(x.text()) for x in hObject.selectedItems()][0]
        if os.path.isdir(sel_item):
            self.layout1_widgets[1].insertItem(0, QIcon(pyIcons.fileIO.folder), sel_item)
            self.layout1_widgets[1].setCurrentIndex(0)
            
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
    # window.setWindowIcon(QIcon('/home/serviceBot/icons/python.png'))
    window.show()
    sys.exit(app.exec_())
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()

# %%
