#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
coco_utils.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/10/30
'''
# =================================================================================================================
# IMPORT PACKAGES
import os, sys
import numpy as np
import logging
from tqdm import tqdm
# FileIO
import json
from pathlib import Path
# Image processing
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from PIL import ImagePath as PILImagePath
import skimage.io as skio
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# COCO tools
from pycocotools import mask
from pycocotools.coco import COCO
# utils
from datetime import datetime
import time
from easydict import EasyDict as edict
import numpy as np
import random
# Custom packages
user_dir = os.path.expanduser('~')
project_dir = Path(user_dir) / 'serviceBot'
sys.path.append(project_dir)
from utils import get_varargin, timeit
import logger_utils
logger_utils.logging_setup()
# =================================================================================================================
def create_coco_info(**kwargs):
    """
    Create info dict that follow coco dataset format
    Ref: http://cocodataset.org/#format-data
    Input Options:
        description: -str. description of the dataset. Default: 'Custom COCO dataset'
        url : -str. url to dataset. Default: ''
        version: -str. dataset version. Default: '1.0'
        year: -int. Default: this year
        contributor: -str. Default: 'author'
        date_created: -str. Default: today()
    Returns:
        coco_info: -edict. Dictonary of coco information
    """
    description = get_varargin(kwargs, 'description', 'Custom COCO dataset')
    url = get_varargin(kwargs, 'url', '')
    version = get_varargin(kwargs, 'version', '1.0')
    year = get_varargin(kwargs, 'year', datetime.today().year)
    contributor = get_varargin(kwargs, 'contributor', 'author')
    date_created = get_varargin(kwargs, 'date_created', datetime.today().strftime('%Y/%m/%d'))
    
    coco_info = edict()
    coco_info.description = description
    coco_info.url = url
    coco_info.version = version
    coco_info.year = year
    coco_info.contributor = contributor
    coco_info.date_created = date_created
    return coco_info
# =================================================================================================================
# 
class coco_json_utils(object):
    """
    Utility to process .json file 
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, json_filepath):
        """[summary]
        
        Arguments:
            json_filepath {str} -- full path to .json file
        """
        self.json_filepath = json_filepath
        self.load_json()
# =================================================================================================================
# METHODS
    def load_json(self, **kwargs):
        """
        Load .json file and create json object
        """
        if not os.path.isfile(self.json_filepath):
            logging.error('{} is not exist'.format(self.json_filepath))
            raise FileNotFoundError('{} is not exist'.format(self.json_filepath))
        else:
            self.coco = COCO(self.json_filepath)
            with open(self.json_filepath) as fid:
                logging.info('Open file: {}'.format(self.json_filepath))
                self.json_obj = json.load(fid)
                
    def get_keys(self, **kwargs):
        """Get key list from .json object
        
        Returns:
            key_list -- list of keys in .json file
        """
        key_list = list(self.json_obj.keys())
        self.json_key = key_list
        return key_list
    
    def display_coco_info(self, **kwargs):
        """
        Display key values information in .json file
        Keyword Options:
            key {str}: Input key to display information. Default: 'info'
            id {int}: Item id to display information. Default: None
        """
        key = get_varargin(kwargs, 'key', 'info')
        item_id = get_varargin(kwargs, 'id', None)
        if key == 'info':
            self.coco.info()
        elif key == 'category':
            cat_id = item_id
            cat_list = self.coco.loadCats(self.coco.getCatIds())
            if cat_id is None:
                print('Number of categories: {}'.format(len(cat_list)))
                nms = [cat['name'] for cat in cat_list]
                print('COCO categories: \n{}\n'.format(' '.join(nms)))
                print('Category list in Json file')
                print(cat_list)
            else:
                cat_id_list = [cat['id'] for cat in cat_list]
                if cat_id in cat_id_list:
                    idx = cat_id_list.index(cat_id)
                    print(cat_list[idx])
                else:
                    print('Category ID: {} is not found'.format(cat_id))
        elif key == 'image':
            img_id = item_id
            img_list = self.coco.loadImgs(self.coco.getImgIds())
            if img_id is None:
                print('Number of Images: {}'.format(len(img_list)))
                print(img_list)
            else:
                img_id_list = [img['id'] for img in img_list]
                if img_id in img_id_list:
                    idx = img_id_list.index(img_id)
                    print(img_list[idx])
                else:
                    print('Image ID: {} is not found'.format(img_id))
        else:
            pass
    
    def get_cat_name(self, catIds):
        """
        Get category name by its ID
        
        Arguments:
            catIds {int} -- category ID
        
        Returns:
            cat_name -- Name of coco category with given ID
        """
        coco_cat = self.coco.loadCats(self.coco.getCatIds())
        cat_id_list = [cat['id'] for cat in coco_cat]
        if catIds in cat_id_list:
            idx = cat_id_list.index(catIds)
            return coco_cat[idx]['name']
        else:
            print('Category ID: {} is not found'.format(catIds))
        
    def get_coco_img(self, **kwargs):
        """
        Get coco images information in .json file
        Keyword Options:
            cat_names {list} -- list of category names. Default: None
            nb_image {int} -- Number of input images. Default: 1
        Returns:
            [coco_img] -- list of coco image dictionary in .json file
        """
        cat_names = get_varargin(kwargs, 'cat_names', None)
        nb_img = get_varargin(kwargs, 'nb_image', 1)
        coco_cats = self.coco.getCatIds(catNms = cat_names)
        img_id = self.coco.getImgIds(catIds = coco_cats)
        if nb_img > len(img_id):
            nb_img = len(img_id)
        pick = np.random.choice(len(img_id), size = nb_img, replace=False)
        sel_img_id = [img_id[i] for i in pick]
        coco_img = self.coco.loadImgs(sel_img_id)
        return coco_img
    
    def get_coco_img_id(self, **kwargs):
        """
        Get a list of image ID given category names
        Keyword Options:
            cat_names {list} -- list of category names. Default: None
            nb_image {int} -- number of input images. Default: 1
        Returns:
            [list] -- List of image IDs
        """
        cat_names = get_varargin(kwargs, 'cat_names', None)
        nb_img = get_varargin(kwargs, 'nb_image', 1)
        coco_img_list = self.get_coco_img(cat_names = cat_names, nb_image = nb_img)
        img_id = [coco_img['id'] for coco_img in coco_img_list]
        return img_id
        
    
    def plt_image(self,coco_img_list, **kwargs):
        """
        Plot coco image in .json file. Default image source from images['coco_url']
        Keyword Options:
            show_mask {bool} -- Option to display annotation mask. Default: True
            display_cat {bool} -- Option to display category information. Default: False
        Arguments:
            coco_img_list {[type]} -- [description]
        """
        show_mask = get_varargin(kwargs, 'show_mask', True)
        display_cat = get_varargin(kwargs, 'display_cat', False)
        for img_info in coco_img_list:
            try:
                img = skio.imread(img_info['coco_url'])
            except:
                img = skio.imread(img_info['file_name'])
            plt.imshow(img)
            plt.axis('off')
            if show_mask is True:
                annIds = self.coco.getAnnIds(imgIds = img_info['id'])
                anns = self.coco.loadAnns(annIds)
                self.coco.showAnns(anns)
            plt.show()
            if display_cat is True:
                annIds = self.coco.getAnnIds(imgIds = img_info['id'])
                anns = self.coco.loadAnns(annIds)
                
                
    def export_segmentation_png(self, coco_img_list, **kwargs):
        export_dir = get_varargin(kwargs, 'export_dir', os.getcwd())
        display_opt = get_varargin(kwargs, 'display', True)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
#         imgId_list = [coco_img['id'] for coco_img in coco_img_list]
        filename_list = [os.path.splitext(coco_img['file_name'])[0] for coco_img in coco_img_list]
        for coco_img, filename in tqdm(zip(coco_img_list, filename_list)):
            imgId = coco_img['id']
            pngPath = Path(export_dir) / '{}.png'.format(filename)
            self.cocoSegmentationToPng(self.coco, imgId, pngPath)
            if display_opt is True:    
                plt.figure()
                plt.imshow(skio.imread(pngPath))
                plt.axis('off')
                
    
    def get_key_values(self, **kwargs):
        """
        Get key value from json object
        Options:
            keys: List of key to get value. Default: All key
        """
        key_list = self.get_keys()
        input_keys = get_varargin(kwargs, 'keys', key_list)
        display_opt = get_varargin(kwargs, 'display', True)
        for key in input_keys:
            if key in key_list:
                strcmd = "self.{} = self.json_obj.get('{}')".format(key,key)
                exec(strcmd)                
                if display_opt is True:
                    strcmd = "print(self.{})".format(key)
                    exec(strcmd)
            else:
                logging.info('Key not exist in coco json file: {}'.format(key))
                
# STATIC FUNCTIONS
    @staticmethod
    def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=False, includeCrowd=False):
        '''
        Convert COCO GT or results for a single image to a segmentation map.
        :param coco: an instance of the COCO API (ground-truth or result)
        :param imgId: the id of the COCO image
        :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
        :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
        :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
        '''

        # Init
        curImg = coco.imgs[imgId]
        imageSize = (curImg['height'], curImg['width'])
        labelMap = np.zeros(imageSize)

        # Get annotations of the current image (may be empty)
        imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
        if includeCrowd:
            annIds = coco.getAnnIds(imgIds=imgId)
        else:
            annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)

        # Combine all annotations of this image in labelMap
        #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
        for a in range(0, len(imgAnnots)):
            labelMask = coco.annToMask(imgAnnots[a]) == 1
            #labelMask = labelMasks[:, :, a] == 1
            newLabel = imgAnnots[a]['category_id']

            if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
                raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

            labelMap[labelMask] = newLabel

        return labelMap
    
    @staticmethod
    def cocoSegmentationToPng(coco, imgId, pngPath, includeCrowd=False):
        '''
        Convert COCO GT or results for a single image to a segmentation map and write it to disk.
        :param coco: an instance of the COCO API (ground-truth or result)
        :param imgId: the COCO id of the image (last part of the file name)
        :param pngPath: the path of the .png file
        :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
        :return: None
        '''

        # Create label map
        labelMap = coco_json_utils.cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
        labelMap = labelMap.astype(np.int8)

        # Get color map and convert to PIL's format
        cmap = coco_json_utils.getCMap()
        cmap = (cmap * 255).astype(int)
        padding = np.zeros((256-cmap.shape[0], 3), np.int8)
        cmap = np.vstack((cmap, padding))
        cmap = cmap.reshape((-1))
        assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

        # Write to png file
        png = PILImage.fromarray(labelMap).convert('P')
    #     png.putpalette(cmap) # Luu 191031: Error: Invalid Palette size
        png.save(pngPath, format='PNG')
        
    @staticmethod
    def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
        '''
        Create a color map for the classes in the COCO Stuff Segmentation Challenge.
        :param stuffStartId: (optional) index where stuff classes start
        :param stuffEndId: (optional) index where stuff classes end
        :param cmapName: (optional) Matlab's name of the color map
        :param addThings: (optional) whether to add a color for the 91 thing classes
        :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
        :param addOther: (optional) whether to add a color for the 'other' class
        :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
        '''

        # Get jet color map from Matlab
        labelCount = stuffEndId - stuffStartId + 1
        cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
        cmap = cmapGen(np.arange(labelCount))
        cmap = cmap[:, 0:3]

        # Reduce value/brightness of stuff colors (easier in HSV format)
        cmap = cmap.reshape((-1, 1, 3))
        hsv = matplotlib.colors.rgb_to_hsv(cmap)
        hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
        cmap = matplotlib.colors.hsv_to_rgb(hsv)
        cmap = cmap.reshape((-1, 3))

        # Permute entries to avoid classes with similar name having similar colors
        st0 = np.random.get_state()
        np.random.seed(42)
        perm = np.random.permutation(labelCount)
        np.random.set_state(st0)
        cmap = cmap[perm, :]

        # Add black (or any other) color for each thing class
        if addThings:
            thingsPadding = np.zeros((stuffStartId - 1, 3))
            cmap = np.vstack((thingsPadding, cmap))

        # Add black color for 'unlabeled' class
        if addUnlabeled:
            cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

        # Add yellow/orange color for 'other' class
        if addOther:
            cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

        return cmap
def main():
    json_filepath = Path(user_dir) / 'serviceBot/datasets/via_sample/via_export_coco.json'
    train_imgpath = Path(user_dir) / 'serviceBot/datasets/via_sample/train'
    coco_obj = coco_dataset(json_filepath = json_filepath, imgpath = train_imgpath)
    coco_obj.get_info()
    coco_obj.get_images(display = True)
    
# DEBUG
if __name__ == '__main__':
    main()
    
