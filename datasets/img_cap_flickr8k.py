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
from tqdm import tqdm
#   AI Framework
import tensorflow as tf
import tensorflow.keras as keras # tensorflow 2.0
print('Tensorflow version: {}'.format(tf.__version__))
print('Keras version: {}'.format(keras.__version__))
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.python.keras.engine as KE
import tensorflow.keras.models as KM
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

def load_captions(filepath):
    captions = utils.read_json(filepath = filepath)

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

def make_train_captions(caption_filepath, subsetImg_filepath, **kwargs):
    # Parse input arguments
    save_opt = get_varargin(kwargs, 'save', True)
    data_dir = Path(subsetImg_filepath).parent
    subsetImg_filename = os.path.splitext(Path(subsetImg_filepath).name)[0]
    captions = make_captions(filepath = caption_filepath)
    # Create a set of train image IDs
    doc = utils.read_txt(filepath = subsetImg_filepath)
    subsetid_list = list()
    logging.info('Create Subset Image ID set')
    for line in tqdm(doc.split('\n'), file = logger.logging_tqdm()):
        if len(line) < 1: # skip empty lines
            continue
        identifier = line.split('.')[0] # get the image identifier
        subsetid_list.append(identifier)
        
    subsetid_set = set(subsetid_list)
    
    logging.info('Create Captions for subset image Ids')
    subset_captions = dict()
    for key,val in captions.items():
        if key in subsetid_set:
            subset_captions[key] = val
    # Save results
    if save_opt is True:
        utils.save_json(subset_captions, filepath = data_dir / '{}_captions.json'.format(subsetImg_filename))
    return subset_captions
    
    
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# =================================================================================================================
# DEBUG
def main(**kwargs):
    data_dir = Path('/media/phatluu/Ubuntu_D/datasets/Flickr8k')
    caption_filepath = data_dir / 'Flickr8k_text/Flickr8k.token.txt'
    trainid_filepath = data_dir / 'Flickr8k_text/Flickr_8k.trainImages.txt'
    make_train_captions(caption_filepath, trainid_filepath)
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
