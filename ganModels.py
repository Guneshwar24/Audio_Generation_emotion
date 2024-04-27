from keras import models
from keras.models import Sequential
from keras.layers import (Input, Flatten, Dropout, BatchNormalization, Reshape,
                          Conv1D, Concatenate, Dense, LeakyReLU, UpSampling1D, )

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

def modified_generator(noise_dim, features_dim, output_shape):
    model = models.Sequential()
    # Combine noise and encoded features
    model.add(Concatenate())
    initial_size = output_shape // 16
    depth = 256
    model.add(Dense(initial_size * depth))
    model.add(Reshape((initial_size, depth)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.1))

    # Upsampling blocks
    for _ in range(4):
        model.add(UpSampling1D(size=2))
        depth //= 2
        model.add(Conv1D(depth, kernel_size=25, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.1))

    model.add(Conv1D(1, kernel_size=25, padding='same', activation='tanh'))
    model.add(Reshape((output_shape, 1)))
    return model

# Discriminator Model
def discriminator(InputShape):
    depth = 64

    model = models.Sequential()
    model.add(Input(shape=(InputShape,1)))

    for _ in range(4):  
        depth *= 2  # Increase depth depth
        model.add(Conv1D(depth, kernel_size=3,strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.1))

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
    depth=64
    
    model = Sequential()
    model.add(Input(shape=(InputShape, 1)))

    for _ in range(4):  
        depth *= 2  # Increase depth depth
        model.add(Conv1D(depth, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.1))    
    
    model.add(Flatten())
    model.add(Dense(EncodeSize, activation='relu'))
    return model

# AutoEndoder
def autoEncoder(Encoder, Generator):
    model = Sequential()
    model.add(Encoder)
    model.add(Generator)
    return model