# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:32:04 2019

@author: 212769892
"""
# =============================================================================
# IMPORT PACKAGES
import inspect, os
import time
import logging
import pkgutil
import pandas as pd
import tqdm
from pathlib import Path
# =============================================================================
def get_varargin(kwargs, inputkey, defaultValue):
    outputVal = defaultValue
    for key, value in kwargs.items():
        if key == inputkey:
            outputVal = value
        else:
            pass
    return outputVal
# =================================================================================================================
# DECORATOR
# timeit
# Decorator
def timeit(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        elapsed_time = (time.time() - start_time)*1000
        msg = 'DONE: {func_name}.\t' \
            'Elapsed Time: {elapsed_time:.2f}ms\t'.format(
            func_name = method.__name__,
            elapsed_time = elapsed_time)
        logging.info(msg)
        return result
    return timed
# =================================================================================================================
# FILE IO
@timeit
def unzip_file(filename, **kwargs):
    '''
    unzip file
    '''
    output_dir = get_varargin(kwargs, 'output', os.path.dirname(filename))
    del_zip = get_varargin(kwargs,'remove', True)
    import zipfile
    with zipfile.ZipFile(filename, 'r') as fid:
        fid.extractall(output_dir)
    if del_zip is True:
        os.remove(filename)
# 
def makedir(inputDir):
    if not os.path.exists(inputDir):
        logging.info('Making directory: {}'.format(os.path.abspath(inputDir)))
        os.makedirs(inputDir)
    else:
        logging.info('Directory already exist: {}'.format(os.path.abspath(inputDir)))


# =================================================================================================================
def list_modules(input_package, **kwargs):
    mlist = [name for _,name,_ in pkgutil.iter_modules([os.path.dirname(input_package.__file__)])]
    display_opt = get_varargin(kwargs, 'display', True)
    if display_opt is True:
        for m in mlist:
            print(m)
    return mlist

    
def get_obj_params(obj):
    '''
    Get names and values of all parameters in `obj`'s __init__
    '''
    try:
        # get names of every variable in the argument
        args = inspect.getargspec(obj.__init__)[0]
        args.pop(0)   # remove "self"

        # get values for each of the above in the object
        argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
        return argdict
    except:
        raise ValueError("object has no __init__ method")

def show_obj_params(obj):
    for key, val in zip(obj.__dict__.keys(), obj.__dict__.values()):
        print('{} : {} '.format(key, val))


def display_dict(input):
    for key, val in input.items():
        print(key, ":", val)

def print_ndarray(input_mat):
    """Print ndarray in python as matrix in Matlab
    
    Args:
        input_mat ([type]): [description]
    """
    print('\n'.join(['\t'.join(['{:.1f}'.format(item) for item in row]) for row in input_mat]))

import pickle
def pickle_data(list_of_var, **kwargs):
    filename = get_varargin(kwargs, 'filename', 'untitled.pickle')
    with open(filename, 'wb') as fid:
        for var in list_of_var:
            pickle.dump(var, fid)
# =================================================================================================================
# DEBUG
if __name__ == '__main__':
    logging.info('Hello')