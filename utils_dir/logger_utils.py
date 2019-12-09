#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
logger_utils.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/28
'''
#%%
# =================================================================================================================
# IMPORT PACKAGES
from __future__ import print_function

import os
import io
import inspect, sys
import platform
from subprocess import Popen, PIPE, check_output, STDOUT
# FileIO
import json
from pathlib import Path
from io import BytesIO
import base64
# Computations
import math
import random
# Visualization
from vlogging import VisualRecord
# Utils
from distutils import spawn
import time
from datetime import datetime
import logging
import vlogging
import psutil
from PIL import Image as PILImage
#  DL framework
from tensorflow.python.client import device_lib
# =================================================================================================================
# SETUP
#	Working Directories
current_dir = os.getcwd()
user_dir = os.path.expanduser('~')
filepath = Path(__file__)
project_dir = os.path.abspath(filepath.parents[1])
sys.path.append(project_dir)
#	Custom packages
from utils_dir.utils import timeit, get_varargin
import utils_dir.utils as utils
# =================================================================================================================    
class colorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # msgformat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    msgformat = '%(asctime)s|%(filename)s:%(lineno)d|%(levelname)s| %(message)s'
    FORMATS = {
        logging.DEBUG: grey + msgformat + reset,
        logging.INFO: grey + msgformat + reset,
        logging.WARNING: yellow + msgformat + reset,
        logging.ERROR: red + msgformat + reset,
        logging.CRITICAL: bold_red + msgformat + reset
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%y%m%d-%I:%M')
        return formatter.format(record)

class html_colorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    black = "<p style='color:black; white-space:pre'>"
    blue = "<p style='color:blue'>"
    orange = "<p style='color:#ff8c00; font-weight:bold'>"
    red = "<p style='color:red; font-weight:bold'>"
    span = "<span style='color: #026440;''>"
    reset = "</p>"
    msgformat = '%(asctime)s|%(filename)s:%(lineno)d|%(levelname)s|'
    FORMATS = {
        logging.DEBUG: blue + msgformat + "\t" + '%(message)s' + reset,
        logging.INFO: black + span +  msgformat + "</span>\t" + '%(message)s' + reset,
        logging.WARNING: orange + msgformat + "\t" + '%(message)s' + reset,
        logging.ERROR: red + msgformat +"\t"+ '%(message)s' +  reset,
        logging.CRITICAL: red + msgformat + "\t" + '%(message)s' + reset
    }
    def format(self, record):
        message = str(record.msg)
        message = "<br />".join(message.split("\n"))
        record.msg = message
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%y%m%d-%I:%M')
        return formatter.format(record)
    
def logging_setup(**kwargs):
    default_logfile = os.path.join(current_dir, 'logging.html')
    log_file = get_varargin(kwargs, 'log_file', default_logfile)
    # 	Logging
    logger = logging.getLogger()
    stream_hdl = logging.StreamHandler(sys.stdout)
    file_hdl = logging.FileHandler(log_file, mode = 'a')
    stream_hdl.setFormatter(colorFormatter())
    logger.addHandler(stream_hdl)
    file_hdl.setFormatter(html_colorFormatter())
    logger.addHandler(file_hdl)
    logger.setLevel(logging.INFO)
    # Only keep one logger
    for h in logger.handlers[:-2]: 
        logger.removeHandler(h)

# =================================================================================================================
# tqdm
class logging_tqdm(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    buf = ''
    def __init__(self):
        super(logging_tqdm, self).__init__()
        self.logger = logging.getLogger()
        self.level = logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)
        
# =================================================================================================================
# log figure to html
def logging_figure(fig, **kwargs):
    desc = get_varargin(kwargs, 'description', '')
    img_size = get_varargin(kwargs, 'img_size', 500)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = desc + "<br/>" + "<img src=\'data:image/png;base64,{}\' style = 'width:{}px;height:{}px'>".format(encoded,img_size,img_size)
    logging.info(html)

# log head of dict
def logging_head(inputvar, **kwargs):
    size = get_varargin(kwargs, 'size', 5)
    if type(inputvar) is dict:
        inputdict = inputvar
        for key in list(inputdict)[:size]:
            logging.info('{}: {}'.format(key, inputdict[key]))
# =================================================================================================================
# LOG HARDWARE
def logging_hardware(**kwargs):
    logging.info('Hardware information:')
    device_list = device_lib.list_local_devices()
    for device in device_list:
        logging.info('\t Device type: {}'.format(device.device_type))
        logging.info('\t Device name: {}'.format(device.name))
        logging.info('\t Memory Limit: {}(GB)'.format(device.memory_limit/1e9))
        logging.info('\t Physical Info: {}'.format(device.physical_device_desc))

class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed)/float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu

def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number

def getGPUs():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi 
        # could not be found from the environment path, 
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"
	
    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([nvidia_smi,"--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    #print(output)
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    #print(lines)
    numDevices = len(lines)-1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        #print(line)
        vals = line.split(', ')
        #print(vals)
        for i in range(12):
            # print(vals[i])
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                uuid = vals[i]
            elif (i == 2):
                gpuUtil = safeFloatCast(vals[i])/100
            elif (i == 3):
                memTotal = safeFloatCast(vals[i])
            elif (i == 4):
                memUsed = safeFloatCast(vals[i])
            elif (i == 5):
                memFree = safeFloatCast(vals[i])
            elif (i == 6):
                driver = vals[i]
            elif (i == 7):
                gpu_name = vals[i]
            elif (i == 8):
                serial = vals[i]
            elif (i == 9):
                display_active = vals[i]
            elif (i == 10):
                display_mode = vals[i]
            elif (i == 11):
                temp_gpu = safeFloatCast(vals[i]);
        GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu))
    return GPUs  # (deviceIds, gpuUtil, memUtil)

def log_nvidia_smi_info():
    try:
        info = check_output(["nvidia-smi"], stderr = STDOUT)
        info = info.decode("utf8")
    except Exception as e:
        info = "Executing nvidia-smi failed: " + str(e)
    logging.info(info.strip())

def logPC_usage():
    cpu_usage = psutil.cpu_percent(percpu=True)
    ram_info = psutil.virtual_memory()
    total_ram = ram_info[0]/1e9
    used_ram = ram_info[3]/1e9
    percent_ram = ram_info[2]
    
    logging.info('CPU Info: Used {}%'.format(cpu_usage))
    logging.info('RAM Info: Used [{:.1f}] / Total[{:.1f}](GB)'\
        '. Percent: {:.1f}%'\
            .format(used_ram, total_ram, percent_ram))
    
def logGPU_usage():
    GPUs = getGPUs()
    logging.info('GPUs Usage Profile:')
    logging.info("----------------------------------------------------------------------------")
    msg = '{:1} {:>10} {:>20} {:>20} {:>20}'.format('ID',
    'Name',
    'GPU_Load(%)',
    'Memory_Usages(%)',
    '[Used]/[Total](MB)')
    logging.info(msg)
    logging.info("============================================================================")
    for gpu in GPUs:
        logging.info('{:1d} {:>10s} {:>15.0f}% {:>15.0f}% {:>15.0f}/{:.0f}MB'\
            .format(gpu.id,
            gpu.name,
            gpu.load*100,
            gpu.memoryUtil*100,
            gpu.memoryUsed,
            gpu.memoryTotal))

def log_json_stats(stats, sort_keys=True):
    # hack to control precision of top-level floats
    stats = {
        k: '{:.6f}'.format(v) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    print('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))
    
def log_train_time(duration):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("Total Training Time: {:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))


#%%
# =================================================================================================================
# DEBUG
def main(**kwargs):
    pass
if __name__ == '__main__':
    main()
    # logging_setup()
    # logging.info('Hello!')
    # # pil_image = PILImage.open('/home/phatluu/serviceBot/Downloads/lenna.jpg')
    # # logging.info(vlogging.VisualRecord(
    # #     "Hello from PIL", pil_image, "This is PIL image", fmt="jpeg"))
    # logging.warning('Warning Hello!\n newline ')
    # logging.error('Error message Hello!')
    # logging_hardware()
    # log_nvidia_smi_info()