# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:32:04 2019

@author: 212769892
"""
# =============================================================================
# IMPORT PACKAGES
import inspect, os
import pkgutil
import pandas as pd
# =============================================================================
def get_varargin(kwargs, inputkey, defaultValue):
    outputVal = defaultValue
    for key, value in kwargs.items():
        if key == inputkey:
            outputVal = value
        else:
            pass
    return outputVal

def list_modules(input_package):
    mlist = [name for _,name,_ in pkgutil.iter_modules([os.path.dirname(input_package.__file__)])]
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
    print('\n'.join(['\t'.join(['{:.1f}'.format(item) for item in row]) 
      for row in input_mat]))


import pickle
def pickle_data(list_of_var, **kwargs):
    filename = get_varargin(kwargs, 'filename', 'untitled.pickle')
    with open(filename, 'wb') as fid:
        for var in list_of_var:
            pickle.dump(var, fid)