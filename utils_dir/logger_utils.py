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
import inspect, sys
import platform
from subprocess import Popen, PIPE, check_output, STDOUT
# FileIO
import json
from pathlib import Path
# Computations
import math
import random
# Utils
from distutils import spawn
import time
from datetime import datetime
import logging
import psutil
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
# =================================================================================================================
# DEF
def logging_setup(**kwargs):
    default_logfile = os.path.join(current_dir, 'logging.log')
    log_file = get_varargin(kwargs, 'log_file', default_logfile)
    # 	Logging
    logger = logging.getLogger()
    stream_hdl = logging.StreamHandler(sys.stdout)
    file_hdl = logging.FileHandler(log_file, mode = 'a')
    formatter = logging.Formatter('%(asctime)s|%(filename)s|%(levelname)s| %(message)s', datefmt='%y%m%d-%I:%M')
    stream_hdl.setFormatter(formatter)
    logger.addHandler(stream_hdl)
    file_hdl.setFormatter(formatter)
    logger.addHandler(file_hdl)
    logger.setLevel(logging.INFO)
    # Only keep one logger
    for h in logger.handlers[:-2]: 
        logger.removeHandler(h)
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
    return info.strip()

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
if __name__ == '__main__':
    logging_setup()
    logging.info('Hello!')
    logging_setup()
    # logging_hardware()
    logGPU_usage()
