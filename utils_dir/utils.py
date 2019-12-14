# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:32:04 2019

@author: 212769892
"""
# =============================================================================
# IMPORT PACKAGES
#%%
import inspect, os
import sys
import pkgutil
# FileIO
from pathlib import Path
import requests
import json
import yaml
import h5py
import pickle

import pandas as pd
#	Utilities
from tqdm import tqdm
import time
from datetime import datetime
import logging
import re # for ProgressBar
import functools
# Visualization
import cv2
import skimage.io as skio
import matplotlib.pyplot as plt
from PIL import Image as PILImage
# =================================================================================================================
# Custom packages
user_dir = os.path.expanduser('~')
script_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
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
    @functools.wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()
        logging.info('START: {}'.format(method.__name__))
        result = method(*args, **kwargs)
        elapsed_time = (time.time() - start_time)
        
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        msg = 'DONE: {func_name}.\t' \
            'Elapsed Time: {hours:0>2}:{mins:0>2}:{secs:0>2}\t'.format(
            func_name = method.__name__,
            hours = int(hours), mins = int(minutes), secs = int(seconds))
        
        logging.info(msg)
        return result
    return timed

# Decorator
def verbose(_func = None, *, desc = None):
    def decorator_verbose(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if desc is not None:
                logging.info(desc)
            filepath = get_varargin(kwargs, 'filepath', None)            
            if filepath is not None:
                logging.info('START: {} -- Filename: {}'.format(func.__name__, filepath))
            else:
                
                logging.info('START: {}'.format(func.__name__))
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logging.error(e)
                return None
            msg = 'DONE: {func_name}.\t' \
                .format(func_name = func.__name__)        
            logging.info(msg)
            return result
        return inner
    if _func is None:
        return decorator_verbose
    else:
        return decorator_verbose(_func)
    
# Decorator
def savefig(_func = None, *, filepath = None):
    def decorator_savefig(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if filepath is None:
                # to_file = Path(inspect.getfile(func)).parent / '{}_{}.png'.format(datetime.now().strftime('%y%m%d%H%M'), func.__name__)
                to_file = Path(project_dir) /'outputs' / '{}_{}.png'.format(datetime.now().strftime('%y%m%d%H%M'), func.__name__)
            else:
                to_file = Path(filepath)
            logging.info('START: savefig: {}.'.format(to_file))
            try:
                result = func(*args, **kwargs)
                plt.savefig(to_file)
            except Exception as e:
                logging.error(e)
                return None
            msg = 'DONE: {func_name}.\t' \
                .format(func_name = func.__name__)        
            logging.info(msg)
            return result
        return inner
    if _func is None:
        return decorator_savefig
    else:
        return decorator_savefig(_func)
# =================================================================================================================
# FILE IO
@verbose
def read_txt(**kwargs):
    filepath = get_varargin(kwargs, 'filepath', None)
    ext = os.path.splitext(Path(filepath).name)[1]
    # ext = Path(filepath).suffix
    output = None
    if ext == '.txt':   
        # open the file as read only
        with open(filepath, 'r') as fid:
            output = fid.read()
    else:
        logging.warning('{} is not .txt'. format(ext))
    return output

@verbose
def save_yaml(inputdata, **kwargs):
    filename = get_varargin(kwargs, 'filepath', './untitiled.yaml')
    with open(filename, 'w') as fid:
        yaml.dump(inputdata, fid)
        
@verbose
def save_json(inputdata, **kwargs):
    """Save dict to .json file
    
    Arguments:
    
        inputdata {[dict]} -- [data so save]
        kwargs:
            filepath: str -- .json filepath. Default: './untitile.json'
            overwrite: bool -- option to overwrite existing file. Default: False
    """
    # Parse input arguments
    filepath = get_varargin(kwargs, 'filepath', './untitiled.json')
    overwrite_opt = get_varargin(kwargs, 'overwrite', False)
    
    if os.path.exists(filepath) and overwrite_opt is False:
            logging.info('File exists: {}. Skip saving'.format(filepath))
            return
    # Save json file
    with open(filepath, 'w') as fid:
        json.dump(inputdata, fid)

@verbose
def read_json(**kwargs):
    """Load .json file
    
    Arguments:
    
        Options:
            filepath: -- str. .json filepath
    
    Returns:
        [dict] -- .json data
    """
    filename = get_varargin(kwargs, 'filepath', './untitiled.json')
    with open(filename, 'r') as fid:
        data = fid.read()
    return data

@verbose
def save_pickle(input_list, **kwargs):
    """Save data to pickle file
    
    Arguments:
    
        input_list {[type]} -- [description]
        kwargs:
            filepath -- str. path to save pickle file. Default. './untitled.pickle'
    """
    filename = get_varargin(kwargs, 'filepath', './untitled.pickle')
    with open(filename, 'wb') as fid:
        for var in input_list:
            pickle.dump(var, fid)

@verbose
def read_pickle(**kwargs):
    filename = get_varargin(kwargs, 'filepath', './untitiled.json')
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)
    return data

@verbose
def read_df(**kwargs):
    """Load .csv or .xlxs file to pandas dataframe
    
    Input:
    
        fullfilename: fullpath to input data file in .csv or .xlsx format
        Options:
            skiprows: row index to skip
            
    Returns:
    
        df: pandas dataframe
    """
    fullfilename = get_varargin(kwargs, 'filepath', None)
    skiprows = get_varargin(kwargs, 'skiprows',None)
    filename,file_ext = os.path.splitext(fullfilename)
    # try:
    if file_ext == '.csv':        
        df = pd.read_csv(fullfilename, skiprows = skiprows)
    else:
        df = pd.read_excel(fullfilename, skiprows = skiprows)    
    return df
    # except Exception as e:
        # logging.error('Failed to load dataframe.\n{}'.format(str(e)))
        
# =================================================================================================================
# COMMENTS
def pickle_data(list_of_var, **kwargs):
    filename = get_varargin(kwargs, 'filename', 'untitled.pickle')
    with open(filename, 'wb') as fid:
        for var in list_of_var:
            pickle.dump(var, fid)

# =================================================================================================================
@timeit
def unzip_file(filename, **kwargs):
    '''
    unzip file
    Options:
        output -- str. Directory path. Default. Same as input filename
        remove -- Boolean. Option to delete zip file. Default. True
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
        logging.info('mkdir: Directory already exist: {}'.format(os.path.abspath(inputDir)))

@verbose
def make_video(imagelist_filepath, **kwargs):
    filepath = get_varargin(kwargs, 'to_file', 'generated_video.avi')
    fps = get_varargin(kwargs, 'fps', 4)
    frame = cv2.imread(imagelist_filepath[0])
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(filepath, 0, fps = fps, frameSize = (width, height))
    # Append image to video
    for image in imagelist_filepath:
        video.write(cv2.imread(image))
    # Clean up
    cv2.destroyAllWindows()
    video.release()
       
# Download file from url
@timeit
def download_url(url, to_file, **kwargs):
    skip_on_avai = get_varargin(kwargs, 'skip_on_avai', True)
    if skip_on_avai is True:
        if os.path.exists(to_file):
            logging.info('File exists: {}. Skip downloading'.format(to_file))
            return -1
    logging.info('Downloading to: {}'.format(to_file))
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    makedir(Path(to_file).parent)
    with open(to_file, 'wb') as fid:
        for data in r.iter_content(block_size):
            t.update(len(data))
            fid.write(data)
    t.close()
    print('\n')

class ProgressBar(object):
    def __init__(self, total,  **kwargs):
        width = get_varargin(kwargs, 'width', 40)
        title = get_varargin(kwargs, 'title', 'Progress')
        symbol = get_varargin(kwargs, 'symbol', '=')
        printer = get_varargin(kwargs, 'printer', 'stdout')
        start = get_varargin(kwargs, 'start', 0)
        
        assert len(symbol) == 1
        fmt = '{}: %(current)03d/%(total)03d %(bar)s  (%(percent)3d%%)'.format(title)
        self.total = total
        self.width = width
        self.symbol = symbol
        self.printer = printer
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)
        self.current = start

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        if self.printer == 'stdout':
            print('\r' + self.fmt % args,  end='')
        else:
            logging.info(self.fmt % args)
        
def list_fulldir(rootdir, **kwargs):
    get_latest = get_varargin(kwargs, 'get_latest', False)
    dirlist = [os.path.join(rootdir, f) for f in os.listdir(rootdir)]
    dirlist = [f for f in dirlist if os.path.isdir(f)]
    if get_latest is True:
        return sorted(dirlist)[-1]
    else:
        return dirlist
    
# Select files from input_directory
def select_files(root_dir, **kwargs):
    """
    Select files in root_directory
    Options:
        and_key -- list. list of keys for AND condition. Default: None
        or_key -- list. List of keys for OR condition. Default: None
        ext -- str. File extension. Default: 'all'
    Returns:
        [list] -- list of selected files
    """
    # Input arguments
    and_key = get_varargin(kwargs, 'and_key', None)
    or_key = get_varargin(kwargs, 'or_key', None)
    sel_ext = get_varargin(kwargs, 'ext', ['all'])
    depth = get_varargin(kwargs, 'depth', 'all')
    # 
    def check_andkeys(filename, and_keys):
        status = True
        if and_keys is not None and not all(key.lower() in filename.lower() for key in and_keys):
            status = False
        return status
    def check_orkeys(filename, or_keys):
        status = True
        if or_keys is not None and not any(key.lower() in filename.lower() for key in or_keys):
            status = False
        return status
    fullfile_list = []
    if depth == 'all':
        for path, subdirs, files in os.walk(root_dir):
            for name in files:
                fullfile_list.append(os.path.join(path, name))
    elif depth == 'root':
        for name in os.listdir(root_dir):
            fullfile_list.append(os.path.join(path, name)) 
    else:
        pass
    sel_files = []
    for fullfile in fullfile_list:        
        # filename, ext = os.path.splitext(os.path.split(fullfile)[1])
        # if set([ext]).issubset(set(sel_ext)) or sel_ext[0] == 'all':
        and_check = check_andkeys(fullfile, and_key)
        or_check = check_orkeys(fullfile, or_key)
        if and_check and or_check:
            sel_files.append(fullfile)
    return sel_files
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

def print_dict(input_dict):
    # print(json.dumps(input_dict, indent = 2))
    for key, val in input.items():
        print(key, ":\n")
        print('\t{}'.format(val))

def print_ndarray(input_mat):
    """Print ndarray in python as matrix in Matlab
    
    Args:
        input_mat ([type]): [description]
    """
    print('\n'.join(['\t'.join(['{:.1f}'.format(item) for item in row]) for row in input_mat]))
            
# =================================================================================================================
# DEBUG
file_list = select_files('/media/phatluu/Ubuntu_D/outputs', and_key = ['mnist', 'png'])
# print(file_list)
make_video(file_list)
#%%
def main(**kwargs):
    pass
    # sel_files = select_files(project_dir, ext = ['.h5', '.yaml'])
    # print(sel_files)
    
if __name__ == '__main__':
    main()
# %%
