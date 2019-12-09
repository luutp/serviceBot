#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
fileIO.py
Description:
-------------------------------------------------------------------------------------------------------------------
Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2019/12/04
'''
# =================================================================================================================
# IMPORT PACKAGES
#%%
from __future__ import print_function
import os
import inspect, sys
import argparse
from pathlib import Path
import string
import logging
from numpy import array
import numpy as np
from tqdm import tqdm
import random
import itertools
#   AI Framework
import tensorflow as tf
from tensorflow import keras # tensorflow 2
# import tensorflow.keras as keras # tensorflow 2.0
print('Tensorflow version: {}'.format(tf.__version__))
print('Keras version: {}'.format(keras.__version__))
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
from tensorflow.python.keras import preprocessing as kerasPreprocess
from tensorflow.python.keras import applications as kerasApp
from tensorflow.python.keras import engine as KE
from tensorflow.python.keras import utils as kerasUtils
# from tensorflow.keras.utils import vis_utils, to_categorical

# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# Visualization
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import skimage.io as skio
plt.rcParams['figure.figsize'] = (5,5)
plt.rcParams['figure.autolayout'] = True
plt.rcParams['lines.linewidth'] = 1.75
plt.rcParams['lines.color'] = 'k'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Custom packages
user_dir = os.path.expanduser('~')
script_dir = Path((__file__)).parents[0]
project_dir = os.path.abspath(Path((__file__)).parents[1])
sys.path.append(project_dir)
from utils_dir.utils import timeit, get_varargin
import utils_dir.utils as utils
import utils_dir.logger_utils as logger
logger.logging_setup()
# =================================================================================================================
# COMMENTS 
# extract descriptions for images
def make_captions(**kwargs):
    filepath = get_varargin(kwargs, 'filepath', None)
    save_opt = get_varargin(kwargs, 'save', True)
    data_dir = Path(filepath).parent
    doc = utils.read_txt(filepath=filepath)
    # Parse captions
    logging.info('Create captions for images')
    captions = dict()
    # tqdm_out = TqdmToLogger(logging.getLogger(),level=logging.INFO)
    for line in tqdm(doc.split('\n'), file = logger.logging_tqdm()):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
		# take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
        image_id = image_id.split('.')[0]
		# convert description tokens back to string
        image_desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
		# create the list if needed
        if image_id not in captions:
            captions[image_id] = list()
        captions[image_id].append(image_desc)
    # Clean captions
    logging.info('Clean captions')
    clean_descriptions(captions)
    if save_opt is True:
        utils.save_json(captions, filepath = data_dir / 'flickr8k_captions.json')
    return captions

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

def make_imgID_set(imgID_filepath):
    # Create a set of train image IDs
    doc = utils.read_txt(filepath = imgID_filepath)
    subsetid_list = list()
    logging.info('Create Subset Image ID set')
    for line in tqdm(doc.split('\n')):
        if len(line) < 1: # skip empty lines
            continue
        identifier = line.split('.')[0] # get the image identifier
        subsetid_list.append(identifier)
    imgID_set = set(subsetid_list)
    return imgID_set

@utils.verbose
def make_subset_captions(caption_filepath, subsetImg_filepath, **kwargs):
    # Parse input arguments
    save_opt = get_varargin(kwargs, 'save', True)
    data_dir = Path(subsetImg_filepath).parent
    # subsetImg_filename = os.path.splitext(Path(subsetImg_filepath).name)[0]
    captions = make_captions(filepath = caption_filepath)
        
    subsetid_set = make_imgID_set(subsetImg_filepath)
    
    logging.info('Create Captions for subset image Ids')
    subset_captions = dict()
    for key,val in captions.items():
        if key in subsetid_set:
            subset_captions[key] = val
    return subset_captions

@utils.verbose
def make_subset_feature(feature_pcklfile, subset_filepath, **kwargs):
    # load pickle file
    features = utils.read_pickle(filepath = feature_pcklfile)
    imgID_set = make_imgID_set(subset_filepath)
    
    subset_features = dict()
    for key,val in tqdm(features.items()):
        if key in imgID_set:
            subset_features[key] = val
    return subset_features

# =================================================================================================================
class vocab_model(object):
    def __init__(self, captions):
        self.captions = captions
        self.captions_list = self.to_lines()
        self.tokenizer = self.create_tokenizer()
        self.vocabulary = self.to_vocabulary()
    
    @property
    def max_length(self):
        lines = self.captions_list
        return max(len(d.split()) for d in lines)
    @property
    def vocabulary_size(self):
        return len(self.tokenizer.word_index)+1
    
    # covert a dictionary of clean descriptions to a list of descriptions
    def to_lines(self):
        all_desc = list()
        for key in self.captions.keys():
            [all_desc.append(d) for d in self.captions[key]]
        return all_desc
    
	# fit a tokenizer given caption descriptions
    def create_tokenizer(self):
        lines = self.captions_list
        tokenizer = kerasPreprocess.text.Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
    
	# convert the loaded descriptions into a vocabulary of words
	# build a list of all description strings
    def to_vocabulary(self):
        all_desc = set()
        for key in self.captions.keys():
            [all_desc.update(d.split()) for d in self.captions[key]]
        return all_desc
     
# create sequences of images, input sequences and output words for an image
@utils.verbose
def create_sequences(captions, features):
    vocab_obj = vocab_model(captions)
    descriptions = vocab_obj.captions
    tokenizer = vocab_obj.tokenizer
    max_length = vocab_obj.max_length
    vocab_size = vocab_obj.vocabulary_size
    
    X1, X2, y = list(), list(), list()
	# walk through each image identifier
    for key, desc_list in tqdm(descriptions.items()):
		# walk through each description for the image
        for desc in desc_list:
			# encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
			# split one sequence into multiple X,y pairs
				# split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
                in_seq = kerasPreprocess.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
                out_seq = kerasUtils.to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
                X1.append(features[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

@timeit
def extract_features(directory):
	# load the model
	model = kerasApp.vgg16.VGG16()
	# re-structure the model
	model.layers.pop()
	model = KM.Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	# print(model.summary())
	# extract features from each photo
	features = dict()
	# for name in tqdm(os.listdir(directory), file = logger.logging_tqdm()):
	for name in tqdm(os.listdir(directory)):
		# load an image from file
		filename = Path(directory) / name
		image = kerasPreprocess.image.load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = kerasPreprocess.image.img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

		# prepare the image for the VGG model
		image = kerasApp.vgg16.preprocess_input(image)
		# get features
		with tf.device('/device:cpu:0'):
			feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		# plt.imshow(feature.reshape(224,224))
	return features
# =================================================================================================================
# DEBUG
def main(**kwargs):
    data_dir = Path('/media/phatluu/Ubuntu_D/datasets/Flickr8k')
    img_dir = data_dir / 'Flickr8k_Dataset'
    caption_filepath = data_dir / 'Flickr8k_text/Flickr8k.token.txt'
    trainid_filepath = data_dir / 'Flickr8k_text/Flickr_8k.trainImages.txt'
    train_captions = make_subset_captions(caption_filepath, trainid_filepath)
    train_features = make_subset_feature(data_dir/'Flickr8k_Dataset.pickle', trainid_filepath)
    # utils.print_dict(train_features)
    X1train, X2train, ytrain = create_sequences(train_captions, train_features)
    train_dataset = tf.data.Dataset.from_tensor_slices(({'X1train':X1train,'X2train':X2train},
                                                        ytrain))
    
    strategy = tf.distribute.MirroredStrategy() 
    BUFFER_SIZE = 1000
    BATCH_SIZE_PER_REPLICA = 32
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset_batch = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # logging.info(caption_list[:10])
    # print(len(caption_list))
    # tokenizer = create_tokenizer(train_captions)
    # vocab_size = len(tokenizer.word_index) + 1
    # print('Vocabulary Size: %d' % vocab_size)
    # print(tokenizer.word_index)
    # features = extract_features(img_dir)
    # utils.save_pickle([features], filepath = data_dir / 'Flickr8k_Dataset.pickle')
    # for idx, (key,val) in enumerate(random.sample(train_captions.items(), k = 4)):
    #     fig = plt.figure()
    #     # ax = fig.add_subplot(3,3,idx+1)
    #     img = PILImage.open(img_dir / '{}.jpg'.format(key))
    #     plt.imshow(img)
    #     cap = '\n'.join(val)
    #     strrep = {'startseq':' ',  'endseq':' '}
    #     for oldstr, newstr in strrep.items():
    #         cap = cap.replace(oldstr, newstr)
    #     fig.text(0.5,1, cap, ha='center', va = 'bottom')
    #     plt.axis('off')
    #     logger.logging_figure(fig, description = 'Flickr8k Training Examples')
    # plt.show()
    # logger.logging_head(train_captions)
    
    # extract features from all images
	
main()
#%%        
    # captions = make_captions(filepath)
	# load descriptions
	# doc = utils.read_txt(filepath=filepath)
	# logging.info(doc[:100])
	# parse descriptions
	# descriptions = load_descriptions(doc)
	# utils.print_dict(descriptions)
# print('Loaded: %d ' % len(descriptions)) 
# # clean descriptions
	# clean_descriptions(descriptions)
# # summarize vocabulary
# vocabulary = to_vocabulary(descriptions)
# print('Vocabulary Size: %d' % len(vocabulary))
# # save to file
	# utils.save_json(descriptions, filepath = data_dir / 'Flickr8k.token.json')
# save_descriptions(descriptions, 'descriptions.txt')
if __name__ == '__main__':
    main()
