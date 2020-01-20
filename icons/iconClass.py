#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
iconClass.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/01/18
'''
# =================================================================================================================
# IMPORT PACKAGES
#%%
from __future__ import print_function
import os
import inspect, sys
import argparse
# =================================================================================================================
def get_varargin(kwargs, inputkey, defaultValue):
    outputVal = defaultValue
    for key, value in kwargs.items():
        if key == inputkey:
            outputVal = value
        else:
            pass
    return outputVal
# =================================================================================================================
# MAIN
#%%
class Attribute(object):
    pass

class iconClass(object):
    # Initialize Class
    def __init__(self, **kwargs):
        icon_dir = get_varargin(kwargs, 'icon_dir', os.path.dirname(os.path.abspath(__file__)))
        self.icon_dir = icon_dir
        self.fileIO = Attribute()
        self.fileIO.folder = os.path.join(icon_dir, 'foldericon.gif')
        self.fileIO.new_folder = os.path.join(icon_dir, 'newfolder.png')
        self.fileIO.open_folder = os.path.join(icon_dir, 'folder-open.png')
        self.fileIO.check_folder = os.path.join(icon_dir, 'plot_validataion_folder.gif')
        self.fileIO.excel = os.path.join(icon_dir, 'excel.png')
        self.fileIO.image = os.path.join(icon_dir, 'image.png')
        self.fileIO.txt = os.path.join(icon_dir, 'notesicon.gif')
        self.fileIO.python = os.path.join(icon_dir, 'python.png')
        self.fileIO.html = os.path.join(icon_dir, 'html.png')
        self.fileIO.ipynb = os.path.join(icon_dir, 'jupyter.png')
        self.fileIO.yaml = os.path.join(icon_dir, 'yaml.png')
        self.fileIO.json = os.path.join(icon_dir, 'json.png')
        
        self.action = Attribute()
        self.action.home= os.path.join(icon_dir, 'Home_16.png')
        self.action.new = os.path.join(icon_dir, 'New_16.png')
        self.action.play = os.path.join(icon_dir, 'Run_16.png')
        self.action.pause = os.path.join(icon_dir, 'Pause_16.png')
        self.action.refresh = os.path.join(icon_dir, 'Refresh_16.png')
        self.action.stop = os.path.join(icon_dir, 'Stop_16.png')
        self.action.build = os.path.join(icon_dir, 'Build_16.png')
        self.action.configure = os.path.join(icon_dir, 'Configure_16.png')
        self.action.cancel = os.path.join(icon_dir, 'Cancel_16.png')
        self.action.save = os.path.join(icon_dir, 'Save_16.png')
        self.action.copy = os.path.join(icon_dir, 'Copy_16.png')
        self.action.cut = os.path.join(icon_dir, 'Cut_16.png')        
        self.action.find = os.path.join(icon_dir, 'find.png')
        self.action.print = os.path.join(icon_dir, 'Print_16.png')
        self.action.download = os.path.join(icon_dir, 'Import_16.png')
        self.action.export = os.path.join(icon_dir, 'Export_16.png')
        self.action.pan = os.path.join(icon_dir, 'Pan_mono_16.png')
        self.action.zoom_in = os.path.join(icon_dir, 'Zoom_In_16.png')
        self.action.zoom_out = os.path.join(icon_dir, 'Zoom_Out_16.png')
        self.action.forward = os.path.join(icon_dir, 'Forward_16.png')
        self.action.back = os.path.join(icon_dir, 'Back_16.png')
        self.action.up = os.path.join(icon_dir, 'Up_16.png')
        self.action.down = os.path.join(icon_dir, 'Down_16.png')
        
        self.status = Attribute()
        self.status.spinner = os.path.join(icon_dir, 'Spinner_16.gif')
        self.status.check = os.path.join(icon_dir, 'Validate_16.gif')
        self.status.error = os.path.join(icon_dir, 'Error_16.png')
        self.status.warning = os.path.join(icon_dir, 'Warn_16.png')
        
        self.object = Attribute()
        self.object.binocular = os.path.join(icon_dir, 'binocularsBlack.png')
        self.object.camera = os.path.join(icon_dir, 'camera.png')
        
        

        
        # self.fileIO.pdf =  os.path.join(icon_dir, 'foldericon.gif')
# =================================================================================================================
    def make_path(self, input):
        return os.path.join(self.icon_dir, input)
    @property
    def prop1(self):
        return self._prop1
    @prop1.setter
    def prop1(self, inputVal):
        self._prop1 = inputVal
# =================================================================================================================
def main(**kwargs):
    icon = iconClass()
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    main()