#%%
import os
import sys

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop

import numpy as np
import random

from tqdm import tqdm
# Visualization
import matplotlib.pyplot as plt
# Custom packages
user_dir = os.path.expanduser('~')
project_dir = os.path.join(user_dir, 'serviceBot')
sys.path.append(project_dir)
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin, ProgressBar
from utils_dir import logger_utils as logger_utils
from model import resnet as resnet_utils
import config.baseConfig as baseConfig
logger = logger_utils.htmlLogger(log_file = './191214_gan_logging.html', mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))
logger.nvidia_smi()

logger.info('Check if eprint in utils is working')
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Preprocessing
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
BUFFER_SIZE = 10000
BATCH_SIZE = 128
# train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# Set the dimensions of the noise
z_dim = 100
# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
# strategy = tf.distribute.MirroredStrategy() 
# with strategy.scope():
g = Sequential(name = 'Generator')
g.add(Dense(256, input_dim=z_dim, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(1024, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(784, activation='sigmoid'))  # Values between 0 and 1
g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

d = Sequential(name = 'Discriminator')
d.add(Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(1, activation='sigmoid'))  # Values between 0 and 1
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# g.summary(print_fn = logger.filelogger.info)
# d.summary(print_fn = logger.filelogger.info)
logger.keras_summary(g)

def plot_loss(losses):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]
    #d_acc = [v[1] for v in losses["D"]]
    #g_acc = [v[1] for v in losses["G"]]
    
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    #plt.plot(d_acc, label="Discriminator accuracy")
    #plt.plot(g_acc, label="Generator accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_generated(generator, epoch):
    n_ex = 16
    dim=(4, 4)
    figsize=(6,6)
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(n_ex, 28, 28)

    fig = plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    fig.savefig("outputs/mnist_gan_{:03d}.png".format(epoch))
    logger.log_fig(fig, description = 'Training GAN. epoch = {}'.format(epoch),
                   figsize = (400,400))
    plt.close()
    
# Set up a vector (dict) to store the losses
losses = {"D":[], "G":[]}

@timeit
def train(epochs=1, plt_frq=1):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    nb_samples = X_train.shape[0]
    logger.info('Epochs: {}'.format( epochs))
    logger.info('Batch size: {}'.format(BATCH_SIZE))
    logger.info('Batches per epoch: {}'.format(batchCount))
    for e in tqdm(range(1, epochs+1)):
        idx_shuffle = random.sample(range(nb_samples), nb_samples)
        if e == 1 or e%plt_frq == 0:
            print('-'*15, 'Epoch %d' % e, '-'*15)
        for batch in tqdm(range(batchCount)):  # tqdm_notebook(range(batchCount), leave=False):
            # Create a batch by drawing random index numbers from the training set
            image_batch = X_train[idx_shuffle[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]]
            # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
        # for image_batch in tqdm(train_dataset):
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
            # Generate the images from the noise
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            # Create labels
            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 0.9  # One-sided label smoothing

            # Train discriminator on generated images
            d.trainable = True
            d_loss = d.train_on_batch(X, y)

            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

        # Only store losses from final batch of epoch
        losses["D"].append(d_loss)
        losses["G"].append(g_loss)

        # Update the plots
        if e == 1 or e%plt_frq == 0:
            plot_generated(g,e)
    plot_loss(losses)
    
train(epochs=300, plt_frq=3)
