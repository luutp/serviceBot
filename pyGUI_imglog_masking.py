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

def clone_file(input_filepath, output_filepath, **kwargs):
    # filepath = os.path.join(project_dir, 'pyqt.py')
    # output_filepath = os.path.join(project_dir, 'pyqt_copy.py')
    with open(input_filepath, 'r') as fid:
        output = fid.readlines()
    
    with open(output_filepath, 'w') as fid:
        for line in output:
            fid.write(line)

def matplotlib_canvas(**kwargs):
    layout_type = get_varargin(kwargs, 'layout', 'vertical') 
    if layout_type == 'vertical':
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    figure = Figure()
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(canvas)
    return layout, figure

@log_info
def draw_rectangle(topleft, width, height, ax):
    rect = mpl_patches.Rectangle(topleft, width, height, linewidth = 1, edgecolor = 'r')
    ax.add_patch(rect)
    plt.show()
    return rect

class polygonSelector(object):
    def __init__(self, ax, bg_img, label_id = 128):
        self.canvas = ax.figure.canvas
        # Ensure that we have separate colors for each object
        self.poly = mpl_widgets.PolygonSelector(ax, self.onselect,
                                               lineprops = dict(color = 'g', alpha = 1),
                                                markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
        self.label_id = label_id
        self.bg_img = bg_img
        self.mask_img = np.zeros_like(self.bg_img)
        
    def onselect(self, verts):
        self.path = Path(verts)
        self.reset_mask()
        self.mask_img = self.set_mask_label(self.mask_img, self.path, self.label_id)
        plt.imshow(self.mask_img, cmap = 'gray_r')
        self.canvas.draw_idle()
    
    def reset_mask(self):
        self.mask_img = np.zeros_like(self.bg_img)

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()
        
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
        # self.fig.canvas.draw()

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
        elif comp == 'combobox':
            py_comp = QComboBox()            
            py_comp.setInsertPolicy(QComboBox.InsertAtTop)
            py_comp.setEditable(True)
            for ic, txt in zip(icon, label):
                py_comp.addItem(QIcon(ic), txt)
            # QIcon(pyIcons.fileIO.folder), sel_dir)
        elif comp == 'label':
            py_comp = QLabel()
            py_comp.setText(html_text(icon, label))
        elif comp == 'list':
            py_comp = QListWidget()
            py_comp.addItems(label)
            # py_comp.setSelectionMode(QAbstractItemView.MultiSelection)
        elif comp == 'radio':
            py_comp = QRadioButton(label)
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
                                           icons = [pyIcons.fileIO.open_folder, [pyIcons.fileIO.folder]*1],
                                           labels = ['Browse...', [project_dir]],                            
                                           ratios = [20, 80])  
        self.layout2, self.layout2_widgets = uigridcomp(components = ['list'], 
                                           icons = [None],
                                           labels = [['List Item 1', 'List Item 2', 'List Item 3']])  
        self.layout3, self.layout3_widgets = uigridcomp(components = ['edit', 'combobox', 'pushbutton'], 
                                           icons = [None, 
                                                    [pyIcons.fileIO.python, pyIcons.fileIO.excel, 
                                                     pyIcons.fileIO.txt, pyIcons.fileIO.html, None], 
                                                    pyIcons.action.download],
                                           labels = [None, ['.py', '.cvs' , '.txt', '.html', 'all'], 'Load'],
                                           groupbox = 'Selected File',
                                           ratios = [60, 20,20])
        
        self.layout4, self.fig = matplotlib_canvas(layout = 'vertical')
        
        self.ax = self.fig.add_subplot(111)
        
        self.layout5, self.layout5_widgets = uigridcomp(components = ['label','hSlider', 'edit','label',
                                                                      'label', 'hSlider', 'edit'],
                                                        icons=[pyIcons.action.play, None, None, None,
                                                               pyIcons.action.stop, None, None],
                                                        labels = ['Start', None, None, None,
                                                                  'Stop', None, None],
                                                        groupbox = 'Depth(ft)',
                                                        ratios = [5, 25,10, 10, 5, 25, 10])
        
        self.layout13, self.layout13_widgets = uigridcomp(components = ['pushbutton', 'pushbutton', 'pushbutton', 'pushbutton'], 
                                                          icons = [pyIcons.action.up, pyIcons.action.down, pyIcons.action.cancel, pyIcons.action.new],
                                                          labels = [None, None, 'Remove', 'Add'],
                                                          ratios = [20,20,20,20])  
        self.layout23, self.layout23_widgets = uigridcomp(components = ['combobox'],     
                                                          icons = [[None]*2],                                                      
                                                          labels = [['process1', 'process2']])
        self.layout33, self.layout33_widgets = uigridcomp(components = ['list'],                                                           
                                                          labels = [['List Item 1', 'List Item 2', 'List Item 3']])
        
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
                    
        add_layouts(col1_layout, [self.layout1, self.layout2, self.layout3], 
                    ratios = [30, 60, 10])
        add_layouts(col2_layout, [self.layout4, self.layout5], ratios = [90, 10])
        add_layouts(col3_layout, [self.layout13, self.layout23, self.layout33], ratios = [20,20, 60])
        add_layouts(main_layout, [col1_layout, col2_layout, col3_layout], 
                    ratios = [20, 60, 20])
        # self.setLayout(main_layout)    
        self.centralWidget().setLayout(main_layout)   
    
    def set_main_window(self):
        # Window settings
        self.setGeometry(2000, 100, 1400, 900)
        self.setWindowTitle("PyGUI Image Log Processing")
        self.setStyleSheet("background-color: palette(window)")
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
        # self.ax.plot(time, np.sin(time))
        blank_img = np.zeros((720, 360), dtype = np.uint8)
        self.ax.imshow(blank_img)
        # self.circle = mpl_patches.Circle([100, 100], radius = 50, color = 'b', fill = False)
        # #use add_patch instead, it's more clear what you are doing
        # self.ax.add_patch(self.circle)
        self.ax.set_aspect('equal')
        
        # circles = [mpl.patches.Circle((100, 100), 50, fc='r', alpha=0.5, picker = 5),
        #            mpl.patches.Circle((50, 50), 50, fc='b', alpha=0.5, picker = 5)]
        # for circ in circles:
        #     self.ax.add_patch(circ)
        self.ax.add_patch(mpl.patches.Circle((100, 100), 50, fc='r', alpha=0.5, picker = 5))
        self.ax.add_patch(mpl.patches.Rectangle((50, 50), 100, 50, fc='b', alpha = 0.3, picker = 5))
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.poly = mpl_widgets.PolygonSelector(self.ax,
                                    self.onselect,
                                    lineprops = dict(color = 'g', alpha = 1),
                                    markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
        self.path = None
        self.polypatches = None
        # self.polypatches = mpl.patches.Polygon(np.array([[100, 300],[150, 350],[50, 400],[100, 300]]), fc = 'g', alpha = 0.3, picker = 5)
        # self.ax.add_patch(mpl.patches.Polygon(np.array([[100, 300],[150, 350],[50, 400],[100, 300]]), fc = 'g', alpha = 0.3, picker = 5))
        self.current_artist = None
        self.currently_dragging = False
        plt.show()
        
    def onselect(self, verts):
        path = Path(verts)
        self.fig.canvas.draw_idle()
        # self.path = path
        # self.ax.add_patch(mpl.patches.Polygon(np.asarray(self.path.vertices), fc = 'g', alpha = 0.5, picker = 5))
        # self.path.remove()
        # self.poly.disconnect_events()
        
    def on_press(self, event):
        pass

    def on_release(self, event):
        # pass
        self.currently_dragging = False
        self.current_artist = None

    def on_pick(self, event):
        self.currently_dragging = True
        if self.current_artist is None:
            self.current_artist = event.artist
        logger.info("Bedding selected: {}".format(self.current_artist))
        x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
        x0, y0 = x1,y1
        if self.current_artist.__class__.__name__.lower() == 'circle':
            x0, y0 = event.artist.center
        elif  self.current_artist.__class__.__name__.lower() == 'rectangle':
            x0, y0 = event.artist.xy
        elif  self.current_artist.__class__.__name__.lower() == 'polygon':
            x0, y0 = event.artist.xy[0][0], event.artist.xy[0][1]
        
        self.offset = (x0 - x1), (y0 - y1)
        print(x0,y0)
        print(self.offset)

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return
        dx, dy = self.offset
        if self.current_artist.__class__.__name__.lower() == 'circle':
            self.current_artist.center = event.xdata + dx, event.ydata + dy
        elif self.current_artist.__class__.__name__.lower() == 'rectangle':
            self.current_artist.xy = event.xdata + dx, event.ydata + dy
        elif self.current_artist.__class__.__name__.lower() == 'polygon':
            new_xy = np.array([])
            for xy in self.current_artist.xy:
                displacement =  np.array([event.xdata + dx, event.ydata + dy])
                xy =  xy - self.current_artist.xy[0] + displacement
                new_xy = np.vstack((new_xy, xy)) if new_xy.size else xy
            self.current_artist.xy = new_xy
        self.current_artist.figure.canvas.draw()
        
    # Define callback functions
    def define_callback(self):
        self.layout1_widgets[0].clicked.connect(lambda:self.pushbutton_browse_callback(self.layout1_widgets[0]))
        self.layout1_widgets[1].currentIndexChanged.connect(lambda:self.combobox_browse_callback(self.layout1_widgets[1]))
        self.layout2_widgets[0].doubleClicked.connect(lambda:self.listbox_browse_doubleclicked_callback(self.layout2_widgets[0]))
        self.layout2_widgets[0].clicked.connect(lambda:self.listbox_browse_clicked_callback(self.layout2_widgets[0]))
        self.layout3_widgets[1].currentIndexChanged.connect(lambda:self.combobox_ext_callback(self.layout3_widgets[1]))
        self.layout3_widgets[2].clicked.connect(lambda:self.pushbutton_save_callback(self.layout3_widgets[2]))
        self.layout13_widgets[0].clicked.connect(lambda:self.pushbutton_moveup_callback(self.layout13_widgets[0]))
        self.layout13_widgets[1].clicked.connect(lambda:self.pushbutton_movedown_callback(self.layout13_widgets[1]))
        self.layout13_widgets[2].clicked.connect(lambda:self.pushbutton_remove_callback(self.layout13_widgets[2]))
        self.layout13_widgets[3].clicked.connect(lambda:self.pushbutton_add_callback(self.layout13_widgets[3]))
        
        # self.fig.canvas.mpl_connect('button_press_event', self.mpl_button_press_callback)
        # self.fig.canvas.mpl_connect('button_release_event', self.mpl_button_release_callback)
        # self.fig.canvas.mpl_connect('motion_notify_event', self.mpl_motion_notify_callback)
        # self.layout3_widgets[0].valueChanged.connect(lambda:self.slider_edit_callback(self.layout3_widgets[0]))
        # self.layout3_widgets[1].editingFinished.connect(lambda:self.edit_slider_callback(self.layout3_widgets[1]))
    # =================================================================================================================
    # CALLBACKS
    # def mpl_button_press_callback(self, event):
    #     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #         ('double' if event.dblclick else 'single', event.button,
    #         event.x, event.y, event.xdata, event.ydata))
        # art = mpl_patches.Circle([event.xdata, event.ydata], radius = 20, color = 'r', fill = False)
        # self.ax.add_patch(art)
        # self.circle.set_center((event.xdata, event.ydata))
        # self.fig.canvas.draw()
  
    
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
    def pushbutton_moveup_callback(self, hObject):
        pass
    
    @log_info
    def pushbutton_movedown_callback(self, hObject):
        pass
    
    @log_info
    def pushbutton_remove_callback(self, hObject):
        listbox = self.layout33_widgets[0]
        # check = listbox.currentIndex().data()
        selected_row = listbox.currentIndex().row()
        listbox.takeItem(selected_row)
    
    @log_info
    def pushbutton_add_callback(self, hObject):
        combo_val = self.layout23_widgets[0].currentText()
        self.layout33_widgets[0].addItem(combo_val)
    
    
    @log_info
    def combobox_browse_callback(self, hObject):
        currentVal = hObject.currentText()
        file_list = utils.select_files(currentVal, depth = 'root')
        self.set_browser_listbox(self.layout2_widgets[0], file_list)
    
    def set_browser_listbox(self, listbox, file_list):
        listbox.clear()
        if file_list:
            for f in file_list:
                filename = os.path.split(f)[1]
                item = QListWidgetItem(filename)
                file_type = os.path.splitext(filename)[1]
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
                listbox.addItem(item)
            listbox.setIconSize(QSize(16,16)) 
        
    @log_info
    def combobox_ext_callback(self, hObject):
        current_dir = self.layout1_widgets[1].currentText()
        file_ext = hObject.currentText()
        file_list = utils.select_files(current_dir, ext = [file_ext], depth = 'root')
        self.set_browser_listbox(self.layout2_widgets[0], file_list)
        # self.layout2_widgets[0].clear()
        # for f in file_list:
        #     item = QListWidgetItem(f)
        #     file_type = os.path.splitext(f)[1]
        #     if not file_type:
        #         item.setIcon(QIcon(pyIcons.fileIO.folder))                
        #     elif file_type == '.py':
        #         item.setIcon(QIcon(pyIcons.fileIO.python))
        #     elif file_type == '.html':
        #         item.setIcon(QIcon(pyIcons.fileIO.html))
        #     elif file_type == '.ipynb':
        #         item.setIcon(QIcon(pyIcons.fileIO.ipynb))
        #     elif file_type == '.yaml':
        #         item.setIcon(QIcon(pyIcons.fileIO.yaml))
        #     elif file_type == '.json':
        #         item.setIcon(QIcon(pyIcons.fileIO.json))
        #     elif set([file_type]).issubset(set(['.png', '.jpg', '.svg','.gif'])):
        #         item.setIcon(QIcon(pyIcons.fileIO.image))
        #     elif set([file_type]).issubset(set(['.xlsx', '.csv', '.xls'])):
        #         item.setIcon(QIcon(pyIcons.fileIO.excel))
        #     elif set([file_type]).issubset(set(['.txt', '.md'])):
        #         item.setIcon(QIcon(pyIcons.fileIO.txt))
        #     else:
        #         pass 
        #     self.layout2_widgets[0].addItem(item)
        # self.layout2_widgets[0].setIconSize(QSize(16,16)) 
    
    @log_info
    def listbox_browse_doubleclicked_callback(self, hObject):
        sel_item = [str(x.text()) for x in hObject.selectedItems()][0]
        if os.path.isdir(sel_item):
            self.layout1_widgets[1].insertItem(0, QIcon(pyIcons.fileIO.folder), sel_item)
            self.layout1_widgets[1].setCurrentIndex(0)
    
    @log_info
    def listbox_browse_clicked_callback(self, hObject):
        sel_item = [str(x.text()) for x in hObject.selectedItems()][0]
        if not os.path.isdir(sel_item):
            self.layout3_widgets[0].setText(sel_item)
    
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
