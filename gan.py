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
        
from tensorflow.keras import datasets as kerasDatasets
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import Sequential, Model
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
logger = logger_utils.htmlLogger(log_file = './191214_gan_logging.html', mode = 'w')
logger.info('START -- project dir: {}'.format(project_dir))
logger.nvidia_smi()

# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
(X_train, Y_train), (X_test, Y_test) = kerasDatasets.mnist.load_data()
# Preprocessing
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255
BUFFER_SIZE = 10000
BATCH_SIZE = 128*3

train_images = X_train
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# Set the dimensions of the noise
z_dim = 100
# Optimizer
adam = kerasOptimizers.Adam(lr=0.0002, beta_1=0.5)
# with strategy.scope():
# g = Sequential(name = 'Generator')
# g.add(KL.Dense(256, input_dim=z_dim, activation=LeakyReLU(alpha=0.2)))
# g.add(KL.Dense(512, activation=LeakyReLU(alpha=0.2)))
# g.add(KL.Dense(1024, activation=LeakyReLU(alpha=0.2)))
# g.add(KL.Dense(784, activation='sigmoid'))  # Values between 0 and 1
# # g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# d = Sequential(name = 'Discriminator')
# d.add(KL.Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.2)))
# d.add(KL.Dropout(0.3))
# d.add(KL.Dense(512, activation=LeakyReLU(alpha=0.2)))
# d.add(KL.Dropout(0.3))
# d.add(KL.Dense(256, activation=LeakyReLU(alpha=0.2)))
# d.add(KL.Dropout(0.3))
# d.add(KL.Dense(1, activation='sigmoid'))  # Values between 0 and 1
# d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# d.trainable = False
# inputs = Input(shape=(z_dim, ))
# hidden = g(inputs)
# output = d(hidden)
# gan = Model(inputs, output)
# gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# generator = g
# discriminator = d
# logger.keras_summary(g)

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
    fig.suptitle('Training GAN. Epoch: {}'.format(epoch), fontsize = 16)
    fig.subplots_adjust(top = 0.95)
    # plt.show()
    fig.savefig("/media/phatluu/Ubuntu_D/outputs/gan/191214_mnist_gan_{:03d}.png".format(epoch))
    logger.log_fig(fig,
                   figsize = (400,400))
    plt.close()
    
# Set up a vector (dict) to store the losses
losses = {"D":[], "G":[]}

def make_discriminator_model():
    model = tf.keras.Sequential()
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

def make_generator_model():
    model = tf.keras.Sequential()
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

strategy = tf.distribute.MirroredStrategy() 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
with strategy.scope():
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    
# Define loss
with strategy.scope():
    binary_crossentropy = kerasLosses.BinaryCrossentropy(from_logits = True,
                                                         reduction = kerasLosses.Reduction.NONE)
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.reduce_sum(binary_crossentropy(tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_sum(binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        replica_loss = real_loss + fake_loss
        return replica_loss
        # return tf.nn.compute_average_loss(replica_loss, 
                                        #   global_batch_size = BATCH_SIZE)

    def generator_loss(fake_output):
        replica_loss = tf.reduce_sum(binary_crossentropy(tf.ones_like(fake_output), fake_output))
        return replica_loss
        # return tf.nn.compute_average_loss(replica_loss, 
                                        #   global_batch_size = BATCH_SIZE)

with strategy.scope():
    # Define Optimizer
    discriminator_optimizer = kerasOptimizers.Adam(1e-4)
    generator_optimizer = kerasOptimizers.Adam(1e-4)

# Apply Gradient
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".    

EPOCHS = 100

with strategy.scope():
    def train_step(images):
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

with strategy.scope():
    @tf.function
    def distributed_train_step(dataset_inputs):
        strategy.experimental_run_v2(train_step,
                                    args=(dataset_inputs,))
        
    for epoch in range(EPOCHS):
        logger.info('Training GAN. Epoch: {}'.format(epoch))
        
        for image_batch in tqdm(train_dist_dataset):
            # train_step(image_batch)
            distributed_train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            # generate_and_save_images(generator,
            #                         epoch + 1,
            #                         seed)

            # Save the model every 15 epochs
                # checkpoint.save(file_prefix = checkpoint_prefix)

            # print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        if (epoch) % 2 == 0:
            plot_generated(generator,epoch)
        if epoch == 1:
            logger.nvidia_smi()
            
# train(train_dataset, 200)
    # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                         epochs,
    #                         seed)
# @timeit
# def train(epochs=1, plt_frq=1):
#     batchCount = int(X_train.shape[0] / BATCH_SIZE)
#     nb_samples = X_train.shape[0]
#     logger.info('Epochs: {}'.format( epochs))
#     logger.info('Batch size: {}'.format(BATCH_SIZE))
#     logger.info('Batches per epoch: {}'.format(batchCount))
#     for e in tqdm(range(1, epochs+1)):
#         idx_shuffle = random.sample(range(nb_samples), nb_samples)
#         if e == 1 or e%plt_frq == 0:
#             print('-'*15, 'Epoch %d' % e, '-'*15)
#         for batch in tqdm(range(batchCount)):  # tqdm_notebook(range(batchCount), leave=False):
#             # Create a batch by drawing random index numbers from the training set
#             image_batch = X_train[idx_shuffle[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]]
#             # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
#         # for image_batch in tqdm(train_dataset):
#             # Create noise vectors for the generator
#             noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
#             # Generate the images from the noise
#             generated_images = g.predict(noise)
#             X = np.concatenate((image_batch, generated_images))
#             # Create labels
#             y = np.zeros(2*BATCH_SIZE)
#             y[:BATCH_SIZE] = 0.9  # One-sided label smoothing

#             # Train discriminator on generated images
#             d.trainable = True
#             d_loss = d.train_on_batch(X, y)

#             # Train generator
#             noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
#             y2 = np.ones(BATCH_SIZE)
#             d.trainable = False
#             g_loss = gan.train_on_batch(noise, y2)

#         # Only store losses from final batch of epoch
#         losses["D"].append(d_loss)
#         losses["G"].append(g_loss)

#         # Update the plots
#         if e == 1 or e%plt_frq == 0:
#             plot_generated(g,e)
#     plot_loss(losses)
    
# train(epochs=300, plt_frq=3)
