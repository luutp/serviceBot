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
# =================================================================================================================
# EXTERNAL CLASSES
class beddingManager(object):
    def __init__(self, ax, bg_img):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.cids = []
        self.bedding_list = []
        self.bg_img = bg_img
        self.mask_img = np.zeros_like(self.bg_img)
        self.button_press = False
        self.mouse_move = False
        self.selected_bedding = None
        self.connect_default_events()
    
    def connect_event(self, event, callback):
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)
    
    def connect_default_events(self):
        self.connect_event('button_press_event', self.button_press_callback)
        self.connect_event('motion_notify_event', self.motion_notify_callback)
        self.connect_event('button_release_event', self.button_release_callback)
    
    def disconnect_events(self):
        for c in self.cids:
            self.canvas.mpl_disconnect(c)
    
    def add_bedding(self, bedding):
        self.bedding_list.append(bedding)     
        self.selected_bedding = len(self.bedding_list)  
        self.reset_mask()
    
    def remove_bedding(self, idx):
        bedding = self.bedding_list[idx]
        bedding.disconnect()
        bedding.poly.set_visible(False)
        bedding.text_label.set_text('')
        del(bedding)
        self.bedding_list.pop(idx)
        self.reset_mask()
        self.canvas.draw_idle()
        
    def reset_mask(self):
        self.mask_img = np.zeros_like(self.bg_img)
        self.show_mask()
        
    def update_mask(self):
        self.reset_mask()
        logger.info('Update Mask')
        for bedding in self.bedding_list:
            if bedding.path is not None:
                self.mask_img = self.set_mask_label(self.mask_img, 
                                                    bedding.path, 
                                                    bedding.label_id)
        self.show_mask()
        
    def show_mask(self):
        self.ax.imshow(self.mask_img, cmap = 'gray_r')
    
    # CALLBACKS
    def button_press_callback(self, event):
        self.button_press = True
        if event.button == 1: # Left Button Click
            x, y = np.array([event.xdata]), np.array([event.ydata])
            cursor_point = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
            for idx, bedding in enumerate(self.bedding_list):
                is_selected = False
                if bedding.path is not None:
                    is_selected = bedding.path.contains_points(cursor_point)
                    if is_selected:
                        logger.info("Bedding: {} selected".format(bedding.label_id))
                        self.selected_bedding = idx
                        bedding.poly.set_visible(True)
                        bedding.poly.connect_default_events()
                        bedding.text_label.set_color('r')
                    else:
                        bedding.text_label.set_color('k')
                    
                    # bedding.lineprops['color'] = 'r' # = dict(color = 'r', alpha = 1)
                    # bedding.markerprops['mec'] = 'r' # = dict(mec = 'r', mfc = 'r', alpha = 1)
                    
    def motion_notify_callback(self, event):
        if self.button_press is True:
            self.mouse_move = True
    
    def button_release_callback(self, event):
        if self.button_press is True and self.mouse_move is True:
            self.reset_mask()
            logger.info('Mouse Drag')
        self.button_press = False
        self.mouse_move = False
                
    @staticmethod
    def set_mask_label(current_mask, poly_path, label):
        height, width = current_mask.shape
        x, y = np.meshgrid(range(width), range(height))
        coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        temp = current_mask.flatten()
        temp[np.where(mask)] = label
        updated_mask = temp.reshape(height, width)
        return updated_mask
    
class beddingPolygon(object):
    def __init__(self, ax, label_id = 128):
        self.canvas = ax.figure.canvas
        self.ax = ax
        # Ensure that we have separate colors for each object
        self.poly = mpl_widgets.PolygonSelector(ax, self.onselect,
                                               lineprops = dict(color = 'g', alpha = 1),
                                               markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
        self._label_id = label_id
        self.text_label = None
        self.connect()
        self.path = None
        
    @property
    def isactive(self):
        return self.poly.get_active()
    
    @property
    def label_id(self):
        return self._label_id
    
    @label_id.setter
    def label_id(self, inputVal):
        self._label_id = inputVal
        if self.text_label is not None:
            self.text_label.set_text('id_{}'.format(self.label_id))
        
    def onselect(self, verts):
        self.path = Path(verts)
        if self.text_label is None:
            self.text_label = self.ax.text(self.path.vertices[0][0], self.path.vertices[0][1], 
                                           'id_{}'.format(self.label_id),
                                           color = 'k',
                                           rotation = 30,
                                           verticalalignment = 'bottom')
        else:
            self.text_label.set_position((self.path.vertices[0][0], self.path.vertices[0][1]))
        self.canvas.draw_idle()
    
    def connect(self):
        self.poly.connect_default_events()    
    
    def disconnect(self):
        self.poly.disconnect_events()
        # self.canvas.draw_idle()

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
        elif comp == 'mpl_axes':
            figure = Figure()
            py_comp = FigureCanvasQTAgg(figure)
        if not set([comp]).issubset(set(['mpl_axes'])):
            output_widgets.append(py_comp)
        else:
            ax = figure.add_subplot(111)
            output_widgets.append(ax)
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
        self.layout1 = []
        layout, self.pushbutton_browse,self.edit_filepath, self.pushbutton_loadfile  = \
            uigridcomp(components = ['pushbutton', 'edit', 'pushbutton'], 
                       labels = ['Browse..', '', 'Load'], 
                       icons = [pyIcons.fileIO.open_folder, None,  pyIcons.action.download],
                       layout = 'horizontal',                       
                       ratios = [10,80,10])  
        self.layout1.append(layout)
        
        layout, self.pushbutton_add_bedding,self.pushbutton_remove_bedding,_, \
            self.combobox_bedding_id, self.pushbutton_set_bedding_id, _, self.pushbutton_show_mask, self.pushbutton_export_mask = \
            uigridcomp(components = ['pushbutton', 'pushbutton', 'label', 
                                     'combobox', 'pushbutton', 'label', 'pushbutton', 'pushbutton'], 
                       labels = ['Add', 'Remove', None,
                                 ['1:Bedding', '2:Dip', '3:Fault'], 'Set ID', None, 'Show Mask', 'Export'], 
                       icons = [pyIcons.action.new, pyIcons.action.cancel, None, 
                                None,pyIcons.action.confirm, None, pyIcons.action.eye_show,pyIcons.action.export],
                       layout = 'horizontal',
                       groupbox = 'Control Panel',
                       ratios = [10,10, 20, 20, 10,10, 10, 10])  
        self.layout1.append(layout)
        
        layout, self.ax, self.slider_depth,  = \
            uigridcomp(components = ['mpl_axes', 'vSlider'], 
                       layout = 'horizontal') 
        self.slider_depth.setInvertedAppearance(True) 
        self.layout1.append(layout)
        
        layout, label, self.edit_depth,  = \
            uigridcomp(components = ['label','edit'], 
                       labels = ['Depth', None],
                       layout = 'horizontal',
                       ratios = [92, 8])  
        self.layout1.append(layout)
        label.setAlignment(PyQt5.QtCore.Qt.AlignRight | PyQt5.QtCore.Qt.AlignVCenter)
            
        
        self.set_layout()
        self.define_callback()
        self.initialize()
        self.set_main_window()
    
    def set_layout(self):
        main_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()
        # col2_layout = QVBoxLayout()
        # col3_layout = QVBoxLayout()
        def add_layouts(current_layout, layout_list, **kwargs):
            nb_comps = len(layout_list)
            ratio_list = get_varargin(kwargs, 'ratios',[np.floor(100/nb_comps)]*nb_comps)
            for layout, ratio in zip(layout_list, ratio_list):
                layout_type = layout.__class__.__name__
                if layout_type == 'QGroupBox':
                    current_layout.addWidget(layout, ratio)
                else:
                    current_layout.addLayout(layout, ratio)
                    
        add_layouts(col1_layout, self.layout1, ratios = [10, 10, 70, 10])
        add_layouts(main_layout, [col1_layout], 
                    ratios = [20, 30, 50])
        # self.setLayout(main_layout)    
        self.centralWidget().setLayout(main_layout)   
    
    def set_main_window(self):
        # Window settings
        self.setGeometry(1000, 100, 900, 900)
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
        # Set Slider min_max to image log depth
        bg_img = np.zeros((720, 360), dtype = np.uint8)
        self.slider_depth.setValue(0)
        self.slider_edit_depth_callback(self.slider_depth)
        self.bedding_manager = beddingManager(self.ax, bg_img)
        # self.explorer = None
        # self.combobox_module_callback(self.combobox_module)
        # self.checkbox_attrs.setChecked(True)
        # self.checkbox_methods.setChecked(True)
        # self.checkbox_functions.setChecked(True)
        
    # Define callback functions
    def define_callback(self):
        self.pushbutton_browse.clicked.connect(lambda:self.pushbutton_browse_callback(self.pushbutton_browse))
        self.pushbutton_loadfile.clicked.connect(lambda:self.pushbutton_loadfile_callback(self.pushbutton_loadfile))
        self.pushbutton_add_bedding.clicked.connect(lambda:self.pushbutton_add_bedding_callback(self.pushbutton_loadfile))
        self.pushbutton_remove_bedding.clicked.connect(lambda:self.pushbutton_remove_bedding_callback(self.pushbutton_remove_bedding))
        self.pushbutton_set_bedding_id.clicked.connect(lambda:self.pushbutton_set_bedding_id_callback(self.pushbutton_set_bedding_id))
        self.pushbutton_export_mask.clicked.connect(lambda:self.pushbutton_export_mask_callback(self.pushbutton_export_mask))
        self.pushbutton_show_mask.clicked.connect(lambda:self.pushbutton_show_mask_callback(self.pushbutton_show_mask))
        
        self.slider_depth.valueChanged.connect(lambda:self.slider_edit_depth_callback(self.slider_depth))
        self.edit_depth.editingFinished.connect(lambda:self.edit_slider_depth_callback(self.edit_depth))
        # self.combobox_module.currentIndexChanged.connect(lambda:self.combobox_module_callback(self.combobox_module))
        # self.listbox_class.clicked.connect(lambda:self.listbox_class_clicked_callback(self.listbox_class))
        # self.listbox_class_attrs.clicked.connect(lambda:self.listbox_class_attrs_clicked_callback(self.listbox_class_attrs))
        # self.checkbox_attrs.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_attrs))
        # self.checkbox_methods.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_methods))
        # self.checkbox_functions.stateChanged.connect(lambda:self.checkbox_class_members_callback(self.checkbox_functions))
        
        # self.pushbutton_find_methods.clicked.connect(lambda:self.pushbutton_find_methods_callback(self.pushbutton_find_methods))
        # self.combobox_find_methods.currentIndexChanged.connect(lambda:self.combobox_find_methods_callback(self.combobox_find_methods))
        # self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        # self.layout1_widgets[1].currentIndexChanged.connect(lambda:self.combobox_browse_callback(self.layout1_widgets[1]))
        # self.layout2_widgets[0].clicked.connect(lambda:self.listbox_browse_clicked_callback(self.layout2_widgets[0]))
        # self.layout3_widgets[1].currentIndexChanged.connect(lambda:self.combobox_ext_callback(self.layout3_widgets[1]))
        
        
    # =================================================================================================================
    # CALLBACKS    
    @log_info
    def pushbutton_browse_callback(self, hObject):
        dlg = QFileDialog(self, 'Select Directory', project_dir)
        sel_file = None
        if dlg.exec_():
            sel_file = dlg.selectedFiles()[0]
        if sel_file is not None:            
            self.edit_filepath.setText(sel_file)
    @log_info
    def pushbutton_loadfile_callback(self, hObject):
        pass
    @log_info
    def pushbutton_add_bedding_callback(self, hObject):
        bedding = beddingPolygon(self.ax, label_id = 64)
        self.bedding_manager.add_bedding(bedding)
    @log_info
    def pushbutton_remove_bedding_callback(self, hObject):
        self.bedding_manager.remove_bedding(self.bedding_manager.selected_bedding)
    
    @log_info
    def pushbutton_set_bedding_id_callback(self, hObject):
        val = self.combobox_bedding_id.currentText()
        bedding_id = val.split(':')[0]
        if self.bedding_manager.selected_bedding is not None:
            self.bedding_manager.bedding_list[self.bedding_manager.selected_bedding].label_id = int(bedding_id)
        self.ax.figure.canvas.draw_idle()
    
        
    @log_info
    def slider_edit_depth_callback(self, hObject):
        edit = self.edit_depth
        val = hObject.value()
        edit.setText('{:d}'.format(val))
        
    @log_info
    def edit_slider__depth_callback(self, hObject):
        slider = self.slider_depth
        str_val = hObject.text()
        val = float(str_val)
        if val > slider.maximum():
            show_msg_dialog(type = 'warning', message = 'Input > slider.maximum')
            val = slider.maximum()
        elif val < slider.minimum():
            show_msg_dialog(type = 'warning', message = 'Input < slider.minimum')
            val = slider.minimum()
        slider.setValue(val)
    
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
