#%%
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
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
# =================================================================================================================
# Custom packages
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    
user_dir = os.path.expanduser('~')
project_dir = os.path.join(user_dir, 'serviceBot')
if not (project_dir in sys.path):
    print('sys.append: {}'.format(project_dir))
    sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin
import  utils_dir.utils as utils
import utils_dir.logger_utils as logger_utils


#%%
class beddingPolygon(object):
    def __init__(self, ax, bg_img, label_id = 128):
        self.canvas = ax.figure.canvas
        self.ax = ax
        # Ensure that we have separate colors for each object
        self.poly = mpl_widgets.PolygonSelector(ax, self.onselect,
                                               lineprops = dict(color = 'g', alpha = 1),
                                                markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
        self._label_id = label_id
        self.bg_img = bg_img
        self.mask_img = np.zeros_like(self.bg_img)
        self.text_label = None
        self.connect()
        
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
        self.reset_mask()
        self.mask_img = self.set_mask_label(self.mask_img, self.path, self.label_id)
        plt.imshow(self.mask_img, cmap = 'gray_r')
        if self.text_label is None:
            self.text_label = self.ax.text(self.path.vertices[0][0], poly.path.vertices[0][1], 
                                           'id_{}'.format(self.label_id),
                                          rotation = 30,
                                          verticalalignment = 'bottom')
        else:
            self.text_label.set_position((self.path.vertices[0][0], self.path.vertices[0][1]))
        self.canvas.draw_idle()
    
    def reset_mask(self):
        self.mask_img = np.zeros_like(self.bg_img)
    
    def connect(self):
        self.poly.connect_default_events()
        self.poly.connect_event('button_press_event', self.button_press_callback)
    
    def button_press_callback(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
    
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

fig = plt.figure()
ax = fig.add_subplot(111)
bg_img = np.zeros((720, 360), dtype = np.uint8)
ax.imshow(bg_img)

class keyboardControl(object):
    def __init__(self, ax, bg_img):
        self.ax = ax
        self.bg_img = bg_img
        self.canvas = ax.figure.canvas
    def keyboard_callback(self, event):
        if event.key in ['n']:
            poly = polygonSelector(ax, bg_img)
        if event.key in ['x']:
            poly.disconnect()
bedding = beddingPolygon(ax, bg_img)
kb = keyboardControl(ax, bg_img)        
fig.canvas.mpl_connect('key_press_event', kb.keyboard_callback)