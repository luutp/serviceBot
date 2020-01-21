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
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
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

def html_text(icon_path, text, **kwargs):
    iconw = get_varargin(kwargs, 'icon_width', 16)
    iconh = get_varargin(kwargs, 'icon_height', 16)
    font_color = get_varargin(kwargs, 'font_color', 'black')
    font_size = get_varargin(kwargs, 'font_size', 4)
    html_icon = '<img src="{}" height = {} width = {}>'.format(icon_path, iconw, iconh)
    html_text = '<font color = {} size="{}"> {} </font>'.format(font_color, font_size, text)
    html_label = html_icon + html_text
    return html_label

def clone_file(input_filepath, output_filepath, **kwargs):
    # filepath = os.path.join(project_dir, 'pyqt.py')
    # output_filepath = os.path.join(project_dir, 'pyqt_copy.py')
    with open(input_filepath, 'r') as fid:
        output = fid.readlines()
    
    with open(output_filepath, 'w') as fid:
        for line in output:
            fid.write(line)

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
            for ic, txt in zip(icon, label):
                py_comp.addItem(QIcon(ic), txt)
            # QIcon(pyIcons.fileIO.folder), sel_dir)
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
       
        if not set([comp]).issubset(set(['blank'])):
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

# =================================================================================================================
# Main Window
class Form(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setCentralWidget(QWidget(self))
        
        self.layout1, self.layout1_widgets = uigridcomp(components = ['pushbutton', 'combobox'], 
                                           icons = [pyIcons.fileIO.open_folder, [pyIcons.fileIO.folder]*2],
                                           labels = ['Browse...', ['/home/phatluu', project_dir]],                            
                                           ratios = [20, 80])  
        self.layout2, self.layout2_widgets = uigridcomp(components = ['list'], 
                                           icons = [None],
                                           labels = [['List Item 1', 'List Item 2', 'List Item 3']])  
        self.layout3, self.layout3_widgets = uigridcomp(components = ['edit', 'combobox', 'pushbutton'], 
                                           icons = [None, [pyIcons.fileIO.python]*2, pyIcons.action.save],
                                           labels = [None, ['py', 'pyGUI'], 'Save'],
                                           groupbox = 'Output File',
                                           ratios = [60, 20,20])
        self.layout4 = QVBoxLayout()
        figure = Figure()
        canvas = FigureCanvasQTAgg(figure)
        self.layout4.addWidget(canvas)
        self.layout4_ax = figure.add_subplot(111)
        
        self.set_layout()
        self.define_callback()
        self.initialize()
        self.set_main_window()
    
    def set_layout(self):
        main_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()
        def add_layouts(current_layout, layout_list, **kwargs):
            nb_comps = len(layout_list)
            ratio_list = get_varargin(kwargs, 'ratio',[np.floor(100/nb_comps)]*nb_comps)
            for layout, ratio in zip(layout_list, ratio_list):
                layout_type = layout.__class__.__name__
                if layout_type == 'QGroupBox':
                    current_layout.addWidget(layout, ratio)
                else:
                    current_layout.addLayout(layout, ratio)
        add_layouts(col1_layout, [self.layout1, self.layout2, self.layout3], 
                    ratio = [0.2, 0.6, 0.2])
        add_layouts(main_layout, [col1_layout, self.layout4], 
                    ratio = [40, 60])
        # self.setLayout(main_layout)    
        self.centralWidget().setLayout(main_layout)   
    
    def set_main_window(self):
        # Window settings
        self.setGeometry(4000, 100, 1000, 700)
        self.setWindowTitle("PyGUI Make File Template")
        self.setWindowIcon(QIcon(pyIcons.fileIO.html))
        # Add menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        help_menu = menu_bar.addMenu('Help')
        
        new_action = QAction('&New', self)
        new_action.setShortcut("Ctrl+N")
        save_action = QAction('&Save', self)
        save_action.setShortcut("Ctrl+S")
        
        file_menu.addAction(new_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_preferences_menu = file_menu.addMenu('Preferences')
        file_preferences_menu.addAction('Settings')
        
        help_menu.addAction('About')
        
        menu_bar.triggered.connect(self.menubar_callback)
        
        # Add statusbar 
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)
        statusBar.setMaximumHeight(20)
        label_widget = QLabel()
        label_widget.setTextFormat(Qt.RichText)
        # label_widget.setText(self.html_text(pyIcons.fileIO.python, 'Phat Luu'))
        label_widget.setText('v.1.0.0')
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        statusBar.addWidget(progress_bar)
        statusBar.addPermanentWidget(label_widget)
        # 
        self.show()
        
    def menubar_callback(self, hObject):
        print(hObject.__class__.__name__)
        print(hObject)
        print(hObject.text())
        
    def initialize(self):
        self.combobox_browse_callback(self.layout1_widgets[1])
        self.layout3_widgets[0].setText('untitled.py')
        
        time = np.arange(0,2*3.14,0.1)
        self.layout4_ax.plot(time, np.sin(time))
    # Define callback functions
    def define_callback(self):
        self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        self.layout1_widgets[1].currentIndexChanged.connect(lambda:self.combobox_browse_callback(self.layout1_widgets[1]))
        self.layout2_widgets[0].doubleClicked.connect(lambda:self.listbox_browse_callback(self.layout2_widgets[0]))
        self.layout3_widgets[2].clicked.connect(lambda:self.pushbutton_save_callback(self.layout3_widgets[2]))
        # self.layout3_widgets[0].valueChanged.connect(lambda:self.slider_edit_callback(self.layout3_widgets[0]))
        # self.layout3_widgets[1].editingFinished.connect(lambda:self.edit_slider_callback(self.layout3_widgets[1]))
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
    def pushbutton_save_callback(self, hObject):
        output_filename = self.layout3_widgets[0].text()
        file_type = self.layout3_widgets[1].currentText()
        output_dir = self.layout1_widgets[1].currentText()
        
        template_py = os.path.join(project_dir, 'template_py.py')
        output_filepath = os.path.join(output_dir, output_filename)
        if file_type.lower() == 'py':
            logger.info('Making python template file')
            source_file = template_py
        else:
            pass
        clone_file(source_file, output_filepath)
        self.combobox_browse_callback(self.layout1_widgets[1])
    
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
    # =================================================================================================================
    # STATIC METHODS
    # @staticmethod
    
    
def main():
    app = QApplication(sys.argv)
    window = Form()
    sys.exit(app.exec_())
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()
