#%%
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
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

from tensorflow import keras
from tensorflow.keras import datasets as kerasDatasets
from tensorflow.python.keras.utils import np_utils
# from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras import layers as KL
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import optimizers as kerasOptimizers
from tensorflow.keras import losses as kerasLosses

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
logger = logger_utils.htmlLogger(log_file = './191217_dcgan_logging.html', mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))
logger.nvidia_smi()
# =================================================================================================================
# Load Data
(X_train, Y_train), (X_test, Y_test) = kerasDatasets.mnist.load_data()
# Preprocessing
# X_train = X_train.reshape(60000, 784)
BUFFER_SIZE = 10000
BATCH_SIZE = 256
train_images = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
# train_images = X_train.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
# train_images = train_images.reshape(train_images.shape[0], 784)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# Set the dimensions of the noise
z_dim = 100
# DCGAN
def build_discriminator():
    model = tf.keras.Sequential(name = 'Discriminator')
    model.add(KL.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(KL.LeakyReLU())
    model.add(KL.Dropout(0.3))

    model.add(KL.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(KL.LeakyReLU())
    model.add(KL.Dropout(0.3))

    model.add(KL.Flatten())
    model.add(KL.Dense(1))

    return model

def build_generator():
    model = tf.keras.Sequential(name = 'Generator')
    model.add(KL.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(KL.BatchNormalization())
    model.add(KL.LeakyReLU())

    model.add(KL.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(KL.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(KL.BatchNormalization())
    model.add(KL.LeakyReLU())

    model.add(KL.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(KL.BatchNormalization())
    model.add(KL.LeakyReLU())

    model.add(KL.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

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
    fig.suptitle('Training GAN. Epoch: {}'.format(epoch), fontsize = 16)
    fig.subplots_adjust(top = 0.95)
    # plt.show()
    fig.savefig("/media/phatluu/Ubuntu_D/outputs/gan/191214_mnist_gan_{:03d}.png".format(epoch))
    logger.log_fig(fig,
                   figsize = (400,400))
    plt.close()

binary_crossentropy = kerasLosses.BinaryCrossentropy(from_logits = True)
def discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
def generator_loss(fake_output):
    gen_loss = binary_crossentropy(tf.ones_like(fake_output), fake_output)
    return gen_loss

# Define Optimizer
discriminator_optimizer = kerasOptimizers.Adam(0.0002, 0.5)
generator_optimizer = kerasOptimizers.Adam(0.0002, 0.5)

# Apply Gradient
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".    
@tf.function
def train_step(generator, discriminator, images):
    noise = tf.random.normal([BATCH_SIZE, z_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

@timeit
def train(generator, discriminator, epochs,**kwargs):
    # nb_samples = X_train.shape[0]
    # batchCount = int(X_train.shape[0] / BATCH_SIZE)
    for epoch in tqdm(range(1, epochs+1)):
        start_time = time.time()
        # idx_shuffle = random.sample(range(nb_samples), nb_samples)
        # for batch in tqdm(range(batchCount)):
            # image_batch = X_train[idx_shuffle[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]]
        for image_batch in tqdm(train_dataset):
            train_step(generator, discriminator, image_batch)
            
        _,_,secs = utils.elapsed_time(start_time)
        logger.info('Training GAN. Epochs: {}/{}. Batch size: {}. Elapsed time: {:.1f}s'\
            .format( epoch, epochs, BATCH_SIZE, secs))
        if (epoch) % 2 == 0:
            plot_generated(generator,epoch)
        if epoch == 1:
            logger.nvidia_smi()

generator = build_generator()
discriminator = build_discriminator()
logger.keras_summary(generator)
logger.keras_summary(discriminator)

train(generator, discriminator, 200)         