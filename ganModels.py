import numpy as np
import tensorflow as tf

from keras import losses, models, optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (Layer, Input, Flatten, Dropout, BatchNormalization, Reshape,
                          MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D,
                          Conv2DTranspose, Conv2D, Conv1D, Dense, LeakyReLU, ReLU, SpectralNormalization, Activation,
                          LSTM, SimpleRNNCell, UpSampling1D, MaxPooling2D, Cropping2D )
from keras.initializers import RandomNormal

# Generator Model
def generator(NoiseDim, OutputShape):
    model = Sequential()
    
    # Start with a Dense layer that maps the noise to a smaller feature map
    model.add(Dense(128 * 8 * 8, input_dim=NoiseDim))
    model.add(Reshape((8, 8, 128)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Upscale to 16x16
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Upscale to 32x32
    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Upscale to 64x64
    model.add(Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Upscale to 128x128
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

    # Correct the dimensions to match the desired output shape of (128, 126, 1)
    model.add(Cropping2D(cropping=((0, 0), (1, 1))))

    return model

# Discriminator Model
def discriminator(input_shape):
    model = Sequential()
    
    # Correct the input_shape parameter to fit a 3D shape (height, width, channels)
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# Stacked Generator and Discriminator
def stacked_G_D(Generator, Discriminator):
    # The discriminator's weights are not trainable when stacked onto the generator
    discriminator.trainable = False
    
    model = Sequential()
    model.add(Generator)
    model.add(Discriminator)
    return model

# Encoder
def encoder(InputShape, EncodeSize):
    model = Sequential()

    # Input layer
    model.add(Input(shape=InputShape))

    # First convolutional block with MaxPooling
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third convolutional block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten and dense layer for encoded representation
    model.add(Flatten())
    model.add(Dense(EncodeSize, activation='relu'))
    model.add(Dropout(0.5))

    return model
    return model

# AutoEncoder
def autoEncoder(Encoder, Generator):
    model = Sequential()
    model.add(Encoder)
    model.add(Generator)
    return model
