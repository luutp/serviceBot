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
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#	Visualization Packages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import matplotlib.widgets as mpl_widgets
from  matplotlib.path import Path
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

def html_text(icon_path, text, **kwargs):
    iconw = get_varargin(kwargs, 'icon_width', 16)
    iconh = get_varargin(kwargs, 'icon_height', 16)
    font_color = get_varargin(kwargs, 'font_color', 'black')
    font_size = get_varargin(kwargs, 'font_size', 3)
    html_icon = '<img src="{}" height = {} width = {}>'.format(icon_path, iconw, iconh)
    html_text = '<font color = "{}" size="{}">{}</font>'.format(font_color, font_size, text)
    if icon_path is None:
        if text is None:
            html_label = ''
        else:
            html_label = html_text
    else:        
        if text is None:
            html_label = html_icon
        else:
            html_label = html_icon + html_text
    return html_label

class classExplorer(object):
    def __init__(self, class_name, **kwargs):
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
        # func_members = []
        # for mem in self.members:
        #     if 'function' in str(mem[1]):
        #         func_members.append(mem)
        func_members = inspect.getmembers(self.class_name, inspect.isfunction)
        return func_members
    
    @property
    def method_members(self):
        method_members = inspect.getmembers(self.class_name, inspect.ismethod)
        return method_members

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
    
    def list_method_names(self, **kwargs):
        key = get_varargin(kwargs, 'key', 'all')
        names = []
        for mem in self.method_members:
            if not mem[0].startswith('_'):
                names.append(mem[0])
        return self.match_key(names, key)
        
    def list_private_function_names(self, **kwargs):
        key = get_varargin(kwargs, 'key', 'all')
        func_names = []
        for mem in self.function_members:
            if mem[0].startswith('_'):
                func_names.append(mem[0])
        return self.match_key(func_names, key)
    
    def get_docstring(self, input_name):
        output = ''
        for mem in self.members:
            if input_name == mem[0]:
                try:
                    output = inspect.getsource(mem[1])
                except:
                    pass
        return output
    # STATIC METHODS
    @staticmethod
    def match_key(input_list, key):
        if key == 'all':
            return input_list
        else:
            new_list = [item for item in input_list if key in item]
            return new_list
# =================================================================================================================
# GUI DESIGN
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
            py_comp.setCheckable(True)
            py_comp.toggle()
        elif comp == 'hSlider':
            py_comp = create_slider(type = 'horizontal')
        elif comp == 'vSlider':
            py_comp = create_slider(type = 'vertical')
        elif comp == 'edit':
            py_comp = QLineEdit(label)
        elif comp == 'text':
            py_comp = QTextEdit(label)
        elif comp == 'combobox':
            py_comp = QComboBox()            
            py_comp.setInsertPolicy(QComboBox.InsertAtTop)
            py_comp.setEditable(True)
            if icon is not None:
                for ic, txt in zip(icon, label):
                    py_comp.addItem(QIcon(ic), txt)
            else:
                for txt in label:
                    py_comp.addItem(txt)
        elif comp == 'label':
            py_comp = QLabel()
            py_comp.setText(html_text(icon, label))
        elif comp == 'list':
            py_comp = QListWidget()
            if label is not None:
                py_comp.addItems(label)
            # py_comp.setSelectionMode(QAbstractItemView.MultiSelection)
        elif comp == 'radio':
            py_comp = QRadioButton(label)
            py_comp.setIcon(QIcon(icon))
        elif comp == 'checkbox':
            py_comp = QCheckBox(label)
            py_comp.setIcon(QIcon(icon))
            
        if not set([comp]).issubset(set(['label'])):
            output_widgets.append(py_comp)
        
        py_comp.setFont(font)
        layout.addWidget(py_comp, ratio)
        
    if groupbox is not None:
        groupbox_obj = QGroupBox(groupbox)
        gbox_font = font
        gbox_font.setBold(True)
        groupbox_obj.setFont(gbox_font)
        groupbox_obj.setLayout(layout)
        
        outputs =  [groupbox_obj]
    else:
        outputs =  [layout]
    for w in output_widgets:
        outputs.append(w)    
    return outputs

# =================================================================================================================
# Main Window
class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setCentralWidget(QWidget(self))
        self.layout11, self.combobox_module, self.listbox_class  = \
            uigridcomp(components = ['combobox', 'list'], 
                       labels = [['PyQt5.QtWidgets', 'matplotlib.widgets', 'matplotlib.patches'], None], 
                       layout = 'vertical',
                       ratios = [30, 70])  
        self.layout21, self.checkbox_attrs, self.checkbox_methods, self.checkbox_functions  = \
            uigridcomp(components = ['checkbox', 'checkbox', 'checkbox'], 
                       labels = ['Attributes', 'Methods', 'Functions'], 
                       layout = 'vertical',
                       groupbox = 'Class Members')  
        
        self.layout12, self.combobox_find_methods, self.pushbutton_find_methods  = \
            uigridcomp(components = ['combobox', 'pushbutton'], 
                       labels = [['get','set', 'index', 'item', 'event','is','selected'], 'Find'], 
                       icons = [None, pyIcons.action.find],
                       ratios = [70, 30])  
        
        self.layout22,  self.listbox_class_attrs =\
            uigridcomp(components = ['list']) 
        
        self.layout13, self.textedit_docstring = \
            uigridcomp(components = ['text'])  
        
        self.set_layout()
        self.define_callback()
        self.initialize()
        self.set_main_window()
    
    def set_layout(self):
        main_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()
        col2_layout = QVBoxLayout()
        col3_layout = QVBoxLayout()
        def add_layouts(current_layout, layout_list, **kwargs):
            nb_comps = len(layout_list)
            ratio_list = get_varargin(kwargs, 'ratios',[np.floor(100/nb_comps)]*nb_comps)
            for layout, ratio in zip(layout_list, ratio_list):
                layout_type = layout.__class__.__name__
                if layout_type == 'QGroupBox':
                    current_layout.addWidget(layout, ratio)
                else:
                    current_layout.addLayout(layout, ratio)
                    
        add_layouts(col1_layout, [self.layout11, self.layout21], ratios = [60, 20])
        add_layouts(col2_layout, [self.layout12, self.layout22], ratios = [20,80])
        add_layouts(col3_layout, [self.layout13])
        add_layouts(main_layout, [col1_layout, col2_layout, col3_layout], 
                    ratios = [20, 30, 50])
        # self.setLayout(main_layout)    
        self.centralWidget().setLayout(main_layout)   
    
    def set_main_window(self):
        # Window settings
        self.setGeometry(2000, 100, 1400, 900)
        self.setWindowTitle("PyGUI Module Explore")
        self.setStyleSheet("background-color: palette(window)")
        self.setWindowIcon(QIcon(pyIcons.fileIO.html))
        # Add statusbar 
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)
        statusBar.setMaximumHeight(20)
        label_widget = QLabel()
        label_widget.setTextFormat(Qt.RichText)
        # label_widget.setText(self.html_text(pyIcons.fileIO.python, 'Phat Luu'))
        label_widget.setText('PL200124')
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        statusBar.addWidget(progress_bar)
        statusBar.addPermanentWidget(label_widget)
        # 
        self.show()
        
    def initialize(self):
        self.explorer = None
        self.combobox_module_callback(self.combobox_module)
        self.checkbox_attrs.setChecked(True)
        self.checkbox_methods.setChecked(True)
        self.checkbox_functions.setChecked(True)
        
    # Define callback functions
    def define_callback(self):
        self.combobox_module.currentIndexChanged.connect(lambda:self.combobox_module_callback(self.combobox_module))
        self.listbox_class.clicked.connect(lambda:self.listbox_class_clicked_callback(self.listbox_class))
        self.listbox_class_attrs.clicked.connect(lambda:self.listbox_class_attrs_clicked_callback(self.listbox_class_attrs))
        self.checkbox_attrs.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_attrs))
        self.checkbox_methods.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_methods))
        self.checkbox_functions.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_functions))
        
        self.pushbutton_find_methods.clicked.connect(lambda:self.pushbutton_find_methods_callback(self.pushbutton_find_methods))
        self.combobox_find_methods.currentIndexChanged.connect(lambda:self.combobox_find_methods_callback(self.combobox_find_methods))
        # self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        # self.layout1_widgets[1].currentIndexChanged.connect(lambda:self.combobox_browse_callback(self.layout1_widgets[1]))
        # self.layout2_widgets[0].clicked.connect(lambda:self.listbox_browse_clicked_callback(self.layout2_widgets[0]))
        # self.layout3_widgets[1].currentIndexChanged.connect(lambda:self.combobox_ext_callback(self.layout3_widgets[1]))
        # self.layout3_widgets[2].clicked.connect(lambda:self.pushbutton_save_callback(self.layout3_widgets[2]))
        # self.layout13_widgets[0].clicked.connect(lambda:self.pushbutton_moveup_callback(self.layout13_widgets[0]))
        # self.layout13_widgets[1].clicked.connect(lambda:self.pushbutton_movedown_callback(self.layout13_widgets[1]))
        # self.layout13_widgets[2].clicked.connect(lambda:self.pushbutton_remove_callback(self.layout13_widgets[2]))
        # self.layout13_widgets[3].clicked.connect(lambda:self.pushbutton_add_callback(self.layout13_widgets[3]))
        
    # =================================================================================================================
    # CALLBACKS    
    @log_info
    def pushbutton_find_methods_callback(self, hObject):
        key = self.combobox_find_methods.currentText()
        full_list = self.full_attrs_list
        short_list = self.match_key(full_list, key)
        self.listbox_class_attrs.clear()
        self.listbox_class_attrs.addItems(short_list)
    
    @log_info
    def combobox_find_methods_callback(self, hObject):
        self.pushbutton_find_methods_callback(self.pushbutton_find_methods)
    
    @log_info
    def combobox_module_callback(self, hObject):
        currentVal = hObject.currentText()
        instance = eval(currentVal)
        clsmembers = inspect.getmembers(instance, inspect.isclass)
        cls_name_list = [mem[0] for mem in clsmembers]
        self.listbox_class.clear()
        self.listbox_class.addItems(cls_name_list)
    
    @log_info
    def listbox_class_clicked_callback(self, hObject):
        sel_module = self.combobox_module.currentText()
        sel_class = [str(x.text()) for x in hObject.selectedItems()][0]
        class_obj = eval('{}.{}'.format(sel_module, sel_class))
        self.explorer = classExplorer(class_obj)
        self.checkbox_class_members_callback(self.checkbox_attrs)
        
    @log_info
    def listbox_class_attrs_clicked_callback(self, hObject):
        sel_attrs = [str(x.text()) for x in hObject.selectedItems()][0]
        docstr = self.explorer.get_docstring(sel_attrs)
        self.textedit_docstring.setText(docstr)
        
    @log_info
    def checkbox_class_members_callback(self, hObject):
        show_attrs = self.checkbox_attrs.isChecked()
        show_methods = self.checkbox_methods.isChecked()
        show_functions = self.checkbox_functions.isChecked()
        name_list = []
        if self.explorer is not None:
            if show_attrs:
                for item in self.explorer.list_attrs():
                    name_list.append(item)
            if show_methods:
                for item in self.explorer.list_method_names():
                    name_list.append(item)
            if show_functions:
                for item in self.explorer.list_function_names():
                    name_list.append(item)
        
        self.full_attrs_list = name_list
        self.listbox_class_attrs.clear()
        self.listbox_class_attrs.addItems(name_list)
            
        
    
    # =================================================================================================================
    # STATIC METHODS
    @staticmethod
    def match_key(input_list, key):
        if key == 'all':
            return input_list
        else:
            new_list = [item for item in input_list if key in item.lower()]
            return new_list
    
def main():
    app = QApplication(sys.argv)
    window = mainWindow()
    sys.exit(app.exec_())
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()
