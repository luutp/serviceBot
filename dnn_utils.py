# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:32:04 2019

@author: 212769892
"""
# =============================================================================
# IMPORT PACKAGES
import os
from datetime import datetime
import time
# 
import numpy as np
import pandas as pd
import pickle
# Keras
import keras
from keras import backend as K
from keras import initializers, regularizers
from keras import layers
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Concatenate, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model, model_from_json, Model
from keras.utils.vis_utils import plot_model
# Metrics
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Visualization-Plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import seaborn as sns
# =============================================================================
# Global variable
todaystr = datetime.today().strftime('%Y%m%d')
# =============================================================================
def get_varargin(kwargs, inputkey, defaultValue):
    outputVal = defaultValue
    for key, value in kwargs.items():
        if key == inputkey:
            outputVal = value
        else:
            pass
    return outputVal

def logfile_setup(**kwargs):
    """Create evironment for saving DNN model, weight, history, and checkpoint
    Options: 
    logdir: location to save log_files
    model: Model name
    Returns:
        [dict]: log_file dictionary with keys: model, weight, history, ckpts
    """
    # Defined directories
    default_logdir = 'logModels'
    todaystr = datetime.today().strftime('%Y%m%d')       
    default_modelName = todaystr + '-DNN_model'    

    logdir = get_varargin(kwargs, 'logdir', default_logdir)
    modelName = get_varargin(kwargs, 'model', default_modelName)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_file = {}
    log_file['model'] = os.path.join(logdir,modelName + '-json.json')
    log_file['weight'] = os.path.join(logdir,modelName + '-weights.h5')
    log_file['history'] = os.path.join(logdir,modelName + '-hist.pckl')
    log_file['ckpts'] = os.path.join(logdir,modelName + '-ckpts.h5')
    display(log_file)
    return log_file
    
def traintest_info(X_train, X_test, y_train, y_test):    
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))
#     Plot label distribution
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(121)
    n, bins, patches = plt.hist(x=y_train, bins=30, color='#0504aa',
                                alpha=0.7, rwidth=1)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Train Set')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
    ax = fig.add_subplot(122)
    n, bins, patches = plt.hist(x=y_test, bins=30, color='#0504aa',
                                alpha=0.7, rwidth=1)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Test Set')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# =============================================================================
# TRAIN MODEl
# =============================================================================
def train_model(model, log_files, X_train, X_test, Y_train, Y_test, **kwargs):
    """Train model
    
    Args:
        model ([type]): [description]
        log_files ([type]): Dictionary of log files. Can be generated from logfile_setup function
            model: path to model .json file
            weight: path to weight .h5 file
            history: history .pckl file
            ckpts: checkpoint .h5 file
        X_train ([type]): [description]
        X_test ([type]): [description]
        Y_train ([type]): [description]
        Y_test ([type]): [description]
    Options:
        retrain. True/False. Option to retrain model from history. Default. True
        optimizer. Default: 'rmsprop'
        loss. Default: 'categorical_crossentropy'
        metrics. Default: ['accuracy']
        batch_size. Default: 32
        epochs. Default: 1
    """
    
    # Option to compile model
    optimizer = get_varargin(kwargs, 'optimizer', 'rmsprop')
    loss = get_varargin(kwargs, 'loss', 'categorical_crossentropy')
    metrics = get_varargin(kwargs, 'metrics', ['accuracy'])
    # Training parameters:
    batch_size = get_varargin(kwargs, 'batch_size', 32)
    epochs = get_varargin(kwargs, 'epochs', 1)
    # Log files
    ckptsFile = log_files['ckpts']
    histFile = log_files['history']
    model_filepath = log_files['model']
    weight_filepath = log_files['weight']
    # retrain option
    retrain = get_varargin(kwargs, 'retrain', True)
    if retrain is True and os.path.isfile(model_filepath):
        print('Load pre-trained model')
        model = loadModel(log_files['model'], log_files['weight'])
        history = loadHistory(log_files['history'])
    else:
        print('Train model from scratch')
        history = {}
    checkpoint = ModelCheckpoint(ckptsFile, monitor='loss', 
                             verbose=1, save_best_only=True, 
                             mode='min')
    callbacks_list = [checkpoint]
    # Start Training
    start_time = time.time()    
    # Compile model
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = metrics)
    # Deal with imbalanced training data
    y_train = np.argmax(Y_train, axis = 1)
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train),
                                                     y_train)
    currHist = model.fit(X_train, Y_train, 
                             batch_size = batch_size, 
                             epochs = epochs,
                             verbose = 1, 
                             validation_data = (X_test, Y_test),
                             class_weight = class_weights,
                             callbacks = callbacks_list)#, callbacks=[tensorboard])
    print('Update history file: {}'.format(histFile))
    history = update_history([history, currHist.history])
    saveHistory(histFile, history)    
    duration = time.time() - start_time    
    printTrainingTime(duration)
    # Save model
    print('Save model to file: {}'.format(model_filepath))
    saveModel(model, model_filepath, weight_filepath)
    return model, history
# =============================================================================
# Save and load history
# =============================================================================
def saveHistory(histFile, histInput):   
    """Save training history to file
    
    Args:
        histFile ([type]): [description]
        histInput ([type]): [description]
    """
    f = open(histFile, 'wb')
    pickle.dump(histInput, f)
    f.close()
# =============================================================================
def loadHistory(histInputfile):
    """Load history input file
    
    Args:
        histInputfile ([type]): [description]
    
    Returns:
        [history]: [description]
    """
    f = open(histInputfile, 'rb')
    history = pickle.load(f)
    f.close()    
    return history
# =============================================================================
# Update history
from collections import defaultdict
def update_history(dict_list):
    """Combine a list of history
    history = update_history([history, currentHist.history])
    Returns:
        [type]: [description]
    """
    dd = defaultdict(list)    
    for d in dict_list:
        for key, value in d.items():
            if not hasattr(value, '__iter__'):
                value = (value,)
            [dd[key].append(v) for v in value]
    return dict(dd)
# =============================================================================
# Save and Load model
# =============================================================================
def saveModel(model, model_filepath, weight_filepath):
    """Save model structure to json file and weight to h5 file
    
    Args:
        model: NN model
        model_filepath ([type]): path to save model file in .json format
        weight_filepath ([type]): path to save model weight
    """
    # ==== BEGIN ====
    json_string = model.to_json()
    # Save model architecture in JSON file
    jsonFilename = model_filepath
    open(jsonFilename, 'w').write(json_string)
    # Save weights as HDF5
    weightFilename = weight_filepath
    model.save_weights(weightFilename)
    print('Model structure and weights has been saved')
    print('json: %s' %jsonFilename)
    print('weights: %s' %weightFilename)
    # ==== END ====
# =============================================================================
# Load Model
def loadModel(model_filepath, weight_filepath):    
    """Load pre-trained model from .json model file path and .h5 model weight
    
    Args:
        model_filepath ([type]): filepath to model structure
        weight_filepath ([type]): filepath to model weight
    
    Returns:
        [model]: pre-train NN model
    """
    # Load model architecture from JSON file
    jsonFile = model_filepath
    weightFile = weight_filepath
    model = model_from_json(open(jsonFile).read())
    print('%s file is loaded' % jsonFile)
    # Load model weights from HDF5 file
    model.load_weights(weightFile)
    print('%s file is loaded' % weightFile)
    return model
# =============================================================================
def printTrainingTime(duration):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Training Time: {:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))
# =============================================================================
# PLOTS
# =============================================================================
# Decorator to save image
def save_fig(**kwargs):
    """Decorator to save plot to file
    
    Args:
        Options: 
        save: True/False. Option to save image. Default: True
        figname: str. Figure name. Default: 'YYMMdd-untitled.png'
    Returns:
        [type]: [description]
    """
    save_opt = get_varargin(kwargs, 'save', True)
    figname = get_varargin(kwargs, 'figname', '{}-untitled.png'.format(todaystr))
    def outer(func):
        def inner(*args, **kwargs):
            artist = func(*args)
            if save_opt is True:
                print('Saving figure: {}'.format(figname))            
                plt.savefig(figname, dpi = 500, bbox_inches = 'tight')
        return inner
    return outer

def plt_model_loss(history):
    """Generate plot for model loss and accuracy from training history
    Args:
        history ([type]): [description] 
    """
    # save_opt = get_varargin(kwargs, 'save', False)
    # figname = get_varargin(kwargs, 'figname', '{}-model_loss.png'.format(todaystr))
     
    fig = plt.figure(figsize = (10,5))
    # Loss 
    ax = fig.add_subplot(121)
    plt.plot(history['loss'], color = 'k' ) # Training
    plt.plot(history['val_loss'], color = 'r') # Validation
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # Accuracy
    ax = fig.add_subplot(122)
    plt.plot(history['acc'], color = 'k')
    plt.plot(history['val_acc'], color = 'r')
    # Annotation
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    # if save_opt is True:
    #     fig.savefig(figname, dpi = 500, bbox_inches = 'tight')
# =============================================================================
def plt_confusion_matrix(Y_label, Y_predict):
    """Plot confusion matrix for Y_label vs Y_predict
    
    Args:
        Y_label ([type]): Data label
        Y_predict ([type]): Prediction from model
    """
    save_opt = get_varargin(kwargs, 'save', False)
    figname = get_varargin(kwargs, 'figname', '{}-model_loss.png'.format(todaystr))

    fig = plt.figure(figsize = (6,6))
    y_predict_label = np.argmax(Y_predict, axis = 1)    
    y_test_label = np.argmax(Y_label, axis = 1)    
    cm = confusion_matrix(y_true = y_test_label, 
                        y_pred = y_predict_label)
    sns.heatmap(cm, annot = True, linewidth = 0.2, 
                cmap = sns.color_palette("coolwarm", 10), fmt = 'd')
# =============================================================================
def plt_precision_recall(y_true, probas_pred):
    """Plot the prediction/recall curve
    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    Args:
        y_true : array, shape = [n_samples]
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.

        probas_pred : array, shape = [n_samples]
            Estimated probabilities or decision function.
    """
    precision, recall, thresholds = \
        precision_recall_curve(y_true, \
                               probas_pred)
    average_precision = average_precision_score( \
                        y_true, probas_pred)
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: Avg Precision = {0:.2f}'.format(average_precision))

# 
def plt_roc_curve(y_true, probas_pred):
    fpr, tpr, thresholds = roc_curve(y_true, \
                                     probas_pred)
    areaUnderROC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic: AUC = {0:.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', 'box')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
# =============================================================================
def plt_recall_roc_curve(y_true, probas_pred):
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(121)
    plt_precision_recall(y_true, probas_pred)
    ax = fig.add_subplot(122)
    plt_roc_curve(y_true, probas_pred)
    plt.tight_layout()
