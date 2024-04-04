import numpy as np
import tensorflow as tf

from keras import losses, models, optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (Layer, Input, Flatten, Dropout, BatchNormalization, Reshape,
                          MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D,
                          Conv2DTranspose, Conv2D, Conv1D, Dense, LeakyReLU, ReLU, SpectralNormalization, Activation,
                          LSTM, SimpleRNNCell, UpSampling1D, )
from keras.initializers import RandomNormal

# Generator Model
def generator(NoiseDim, OutputShape):
    
    # Assume depth starts from 256 and we expand it through the network
    depth = 256
    initial_size = OutputShape // 4  # Or adjust based on your layer and upsampling design

    model = models.Sequential()

    # Initial dense layer to expand the noise into a larger, but still compact, representation
    model.add(Dense(initial_size * depth, input_shape=(NoiseDim,)))
    model.add(Reshape((initial_size, depth)))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    # UpSampling and Conv1D layers to expand and shape the waveform
    # Adjust the upsampling and convolution structure according to the specifics of your audio generation
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(depth // 2, kernel_size=25, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    depth //= 2  # Decrease depth after each upsampling

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(depth, kernel_size=25, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    # Final layer that reshapes to the output waveform size
    # The final convolution layer to adjust channel size and fine-tune the details
    model.add(Conv1D(1, kernel_size=25, padding='same', activation='tanh'))

    # Ensure output shape matches the desired audio length
    model.add(Reshape((OutputShape, 1)))
    
    
    # Create a Keras Sequential model
    # model = models.Sequential()

    # # Starting with a Dense layer that reshapes the input noise vector into a specified feature map size
    # initial_size = OutputShape // 4  # Or some factor that makes sense for your architecture
    # depth = 1024  # Initial depth, can be adjusted

    # # Starting with a Dense layer that reshapes the input noise vector
    # model.add(Dense(initial_size * depth, input_shape=(NoiseDim,)))
    # model.add(Reshape((initial_size, depth)))
    # model.add(LeakyReLU(negative_slope=0.2))
    # model.add(BatchNormalization())
    # # Dynamically add upsampling layers based on the OutputShape
    # current_size = initial_size
    # while current_size < OutputShape[0]:
    #     depth //= 2  # Halve the depth with each upsampling step
    #     model.add(Conv2DTranspose(depth, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    #     model.add(LeakyReLU(negative_slope=0.2))
    #     model.add(BatchNormalization())
    #     current_size *= 2  # Each upsampling step doubles the size of the feature map

    # # Add the final transposed convolution layer to match the OutputShape channels
    # model.add(Conv2DTranspose(OutputShape[2], kernel_size=(5, 5), strides=(1, 1), padding='same', activation='tanh'))

    # # Upsampling to 8x8
    # model.add(Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(SpectralNormalization(BatchNormalization()))
    # model.add(LeakyReLU(negative_slope=0.2))

    # # Continue with the pattern...
    # # Upsampling to 16x16
    # model.add(Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(SpectralNormalization(BatchNormalization()))
    # model.add(LeakyReLU(negative_slope=0.2))

    # # Upsampling to 32x32
    # model.add(Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(SpectralNormalization(BatchNormalization()))
    # model.add(LeakyReLU(negative_slope=0.2))

    # # Upsampling to 64x64
    # model.add(Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(SpectralNormalization(BatchNormalization()))
    # model.add(LeakyReLU(negative_slope=0.2))

    # # Upsampling to 128x128
    # model.add(Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(SpectralNormalization(BatchNormalization()))
    # model.add(LeakyReLU(negative_slope=0.2))

    # The final convolutional layer that outputs the generated image
    # Assuming the final image is 128x128x1 (if you're working with grayscale images)
    # model.add(Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    # model.add(Reshape(OutputShape, 1))
    # If the final image is 128x128x3 (RGB Images)
    # model.add(Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

# Discriminator Model
def discriminator(InputShape):
    model = models.Sequential()
    
    model.add(Input(shape=(InputShape,1)))
    # If the input image is 128x128x3 (RGB Images)
    # model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(128, 128, 3)))

    # Assuming the input images are 128x128x1 (grayscale images)
    # The first convolutional layer without batch normalization
    # model.add(Conv2D(64, 3, strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(negative_slope=0.2))
    # model.add(Dropout(0.3))

    # # Second Conv2D layer
    # model.add(Conv2D(128, 3, strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(negative_slope=0.2))
    # model.add(SpectralNormalization())
    # model.add(Dropout(0.3))

    # # Continue with the pattern...
    # # Third Conv2D layer
    # model.add(Conv2D(256, 3, strides=(2, 2), padding='same'))
    # model.add(SpectralNormalization(LeakyReLU(negative_slope=0.2)))
    # model.add(Dropout(0.3))

    # # Fourth Conv2D layer
    # model.add(Conv2D(512, 3, strides=(2, 2), padding='same'))
    # model.add(SpectralNormalization(LeakyReLU(negative_slope=0.2)))
    # model.add(Dropout(0.3))

    # # Fifth Conv2D layer
    # model.add(Conv2D(1024, 3, strides=(2, 2), padding='same'))
    # model.add(SpectralNormalization(LeakyReLU(negative_slope=0.2)))
    # model.add(Dropout(0.3))

    # # Flatten the output layer and add the final dense layer
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))
    
    
    # Replace Conv2D layers with Conv1D for 1D audio processing
    model.add(Conv1D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.1))

    model.add(Conv1D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.1))

    model.add(Conv1D(256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.1))

    model.add(Conv1D(512, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.1))

    model.add(Conv1D(1024, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.1))

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
    model = models.Sequential()
    
    # Define the input layer for 1D audio data
    model.add(Input(shape=(InputShape, 1)))

    # First convolutional layer
    model.add(Conv1D(32, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    # Second convolutional layer
    model.add(Conv1D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))
    # model = models.Sequential()
    # model.add(Input(shape=(InputShape, 1)))

    # # First convolutional layer
    # model.add(Conv1D(32, kernel_size=3, strides= 2, padding='same'))
    # model.add(LeakyReLU(negative_slope=0.2))
    # model.add(BatchNormalization())

    # # Second convolutional layer
    # model.add(Conv1D(64, kernel_size=3, strides= 2, padding='same'))
    # model.add(LeakyReLU(negative_slope=0.2))
    # model.add(BatchNormalization())

    # Third convolutional layer
    model.add(Conv1D(128, kernel_size=3, strides= 2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    # Fourth convolutional layer
    model.add(Conv1D(256, kernel_size=3, strides= 2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))
    
    # Fifth convolutional layer
    model.add(Conv1D(512, kernel_size=3, strides= 2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))
    
    # Flatten the convolutional layer's output to feed it into the dense layer
    model.add(Flatten())

    # Dense layer for the encoded representation
    model.add(Dense(EncodeSize, activation='relu')) # Assuming the encoded size is 100
    
    return model

# AutoEndoder
def autoEncoder(Encoder, Generator):
    model = Sequential()
    model.add(Encoder)
    model.add(Generator)
    return model
