#%%
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import os
import sys

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets as kerasDatasets
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL

from tensorflow.keras import optimizers as kerasOptimizers
from tensorflow.python.keras import utils as kerasUtils

import numpy as np
import random
from tqdm import tqdm
# Display
import logging
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import skimage.io as skio
from IPython.display import Image as displayImage

# Custom packages
user_dir = os.path.expanduser('~')
project_dir = os.path.join(user_dir, 'serviceBot')
sys.path.append(project_dir)
import utils_dir.utils as utils
from utils_dir.utils import timeit, get_varargin, ProgressBar
from utils_dir import logger_utils as logger_utils
from model import resnet as resnet_utils
import config.baseConfig as baseConfig
logger = logger_utils.logging_class(log_file = './logging.html', mode = 'a')
logging.info('START -- project dir: {}'.format(project_dir))

#%%
latent_dim = 100
img_shape = (28,28,1)

def build_generator(**kwargs):
    
    z_noise = KL.Input(shape = (latent_dim,))
    
    x = KL.Dense(256)(z_noise)
    x = KL.LeakyReLU(alpha=0.2)(x)
    
    x = KL.Dense(512)(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    
    x = KL.Dense(1024)(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    
    x = KL.Dense(784, activation='tanh')(x)
    x = KL.Reshape(img_shape)(x)

    gen_model = KM.Model(inputs = z_noise, outputs = x, name = 'Generator')

    return gen_model

def build_discriminator(**kwargs):
    
    img_input = KL.Input(shape = img_shape)
    
    x = KL.Flatten(input_shape = img_shape)(img_input)
    
    x = KL.Dense(512)(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    
    x = KL.Dense(256)(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    
    x = KL.Dense(1, activation='sigmoid')(x)
    
    dis_model = KM.Model(inputs = img_input, outputs = x, name = 'Discriminator')

    return dis_model

def build_gan(**kwargs):
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.trainable = False
    
    z_noise = KL.Input(shape = (latent_dim,))
    gen_img = generator(z_noise)
    gan_output = discriminator(gen_img)

    gan_model = KM.Model(z_noise, gan_output)
    return gan_model

def train(generator, discriminator, gan,
          epochs, batch_size=128, sample_interval=50):

    # Load the dataset
    (X_train, _), (_, _) = kerasDatasets.mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    batchCount = int(X_train.shape[0] / batch_size)
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for i in tqdm(range(batchCount)):
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            # Generate a batch of new images
            gen_imgs = generator.predict(noise)
            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = gan.train_on_batch(noise, valid)
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(generator, epoch)

def sample_images(generator, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("outputs/mnist_gan_{:05d}.png".format(epoch))
    plt.close()
    

generator = build_generator()
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = build_discriminator()
# decision = discriminator(generated_image)
# print (decision)
gan = build_gan()
# gan.summary()
# Compile model
optimizer = kerasOptimizers.Adam(0.0002, 0.5)
generator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
gan.compile(loss = 'binary_crossentropy', optimizer = optimizer)

train(generator, discriminator, gan,
      epochs = 100, batch_size=128, sample_interval=1)

#%%