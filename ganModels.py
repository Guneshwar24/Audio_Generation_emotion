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
# def generator(NoiseDim, OutputShape):
    
#     # Assume depth starts from 256 and we expand it through the network
#     depth = 256
#     initial_size = OutputShape // 16  # Or adjust based on your layer and upsampling design

#     model = models.Sequential()

#     # Initial dense layer to expand the noise into a larger, but still compact, representation
#     model.add(Dense(initial_size * depth, input_shape=(NoiseDim,)))
#     model.add(Reshape((initial_size, depth)))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Dropout(rate=0.1))

#     # UpSampling and Conv1D layers to expand and shape the waveform
#     # Adjust the upsampling and convolution structure according to the specifics of your audio generation
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(depth // 2, kernel_size=25, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Dropout(rate=0.1))

#     depth //= 2  # Decrease depth after each upsampling

#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(depth, kernel_size=25, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Dropout(rate=0.1))

#     # Final layer that reshapes to the output waveform size
#     # The final convolution layer to adjust channel size and fine-tune the details
#     model.add(Conv1D(1, kernel_size=25, padding='same', activation='tanh'))

#     # Ensure output shape matches the desired audio length
#     model.add(Reshape((OutputShape, 1)))
    
    
    
#     return model

def generator(NoiseDim, OutputShape):
    depth = 256
    initial_size = OutputShape // 16
    model = models.Sequential()
    model.add(Dense(initial_size * depth, input_shape=(NoiseDim,)))
    model.add(Reshape((initial_size, depth)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.1))

    # Adding more upsampling and transposed convolutional layers
    for _ in range(4):  # Increase to the number of upsampling you need
        model.add(UpSampling1D(size=2))
        depth //= 2  # Reduce depth
        model.add(Conv1D(depth, kernel_size=25, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.1))
    model.add(Conv1D(1, kernel_size=25, padding='same', activation='tanh'))
    model.add(Reshape((OutputShape, 1)))
    return model

# Discriminator Model
def discriminator(InputShape):
    model = models.Sequential()
    
    model.add(Input(shape=(InputShape,1)))
   
    
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
