from keras.models import Sequential
from keras.layers import (Input, Flatten, Dropout, BatchNormalization, Reshape,
                          Conv1D, Dense, LeakyReLU, UpSampling1D,ReLU )

def generator(NoiseDim, OutputShape):
    """
    This Python function generates a deep convolutional neural network model for a GAN generator with
    specified noise dimension and output shape.
    
    :param NoiseDim: The `NoiseDim` parameter in the `generator` function represents the dimensionality
    of the input noise vector that will be used as the input to the generator model. This noise vector
    is typically sampled from a simple distribution like Gaussian noise and serves as the input for
    generating synthetic data by the generator model
    :param OutputShape: OutputShape refers to the desired shape of the output data that the generator
    model will produce. In the provided code snippet, the generator function is designed to generate
    data with a specific shape specified by the OutputShape parameter. This shape will influence the
    architecture and configuration of the neural network model created by the generator
    :return: The function `generator` returns a Keras Sequential model that generates synthetic data
    based on the input noise dimension and output shape provided. The model consists of several layers
    including Dense, Reshape, BatchNormalization, LeakyReLU, Dropout, UpSampling1D, Conv1D, and an
    output Conv1D layer with tanh activation function. The model is designed to generate synthetic data
    by transforming the
    """
    depth = 256
    initial_size = OutputShape // 16
    model = Sequential()
    model.add(Dense(initial_size * depth, input_shape=(NoiseDim,)))
    model.add(Reshape((initial_size, depth)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.1))

    # Adding more upsampling and transposed convolutional layers
    for _ in range(4): 
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
def discriminator(input_shape):
    """
    The function `discriminator` creates a convolutional neural network model for binary classification
    tasks.
    
    :param input_shape: The `input_shape` parameter in the `discriminator` function represents the shape
    of the input data that will be fed into the discriminator model. In this case, it is a single
    integer value that specifies the length of the input data sequence
    :return: The function `discriminator` returns a Sequential model for a discriminator network used in
    a GAN (Generative Adversarial Network). The model consists of several Conv1D layers with increasing
    depth, followed by activation functions, batch normalization, dropout layers, and a final dense
    layer with a sigmoid activation function.
    """
    depth = 64
    model = Sequential()
    model.add(Input(shape=(input_shape, 1)))  

    # Adding several Conv1D layers with increasing depth
    for i in range(4): 
        model.add(Conv1D(depth * (2 ** i), kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.1))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Stacked Generator and Discriminator
def stacked_G_D(Generator, Discriminator):
    """
    The function `stacked_G_D` creates a stacked model with a generator followed by a discriminator,
    where the discriminator's weights are not trainable.
    
    :param Generator: The `Generator` parameter in the `stacked_G_D` function is typically a neural
    network model that generates fake data, such as images or text, based on random noise or other
    input. This model is responsible for creating data that resembles the real data distribution
    :param Discriminator: The `Discriminator` parameter in the `stacked_G_D` function is typically a
    neural network model that is responsible for distinguishing between real and generated data. It is
    commonly used in Generative Adversarial Networks (GANs) to provide feedback to the generator on how
    well it is generating realistic
    :return: The function `stacked_G_D` returns a Keras Sequential model where the Generator is stacked
    on top of the Discriminator. The Discriminator's weights are set to not be trainable in this stacked
    model.
    """
    # The discriminator's weights are not trainable when stacked onto the generator
    discriminator.trainable = False
    
    model = Sequential()
    model.add(Generator)
    model.add(Discriminator)
    return model

# Encoder
def encoder(InputShape, EncodeSize):
    """
    This function defines an encoder neural network model in Python using Conv1D layers with increasing
    depth, LeakyReLU activation, BatchNormalization, Dropout, Flatten, and a Dense layer with ReLU
    activation.
    
    :param InputShape: The `InputShape` parameter in the `encoder` function represents the shape of the
    input data that will be fed into the model. It specifies the number of time steps in the input
    sequence. For example, if your input data has a shape of (100, 1), where 100 is
    :param EncodeSize: The `EncodeSize` parameter in the `encoder` function represents the size of the
    encoded output or latent space representation that the model will learn to generate. This size
    determines the dimensionality of the compressed representation of the input data after it has been
    processed by the convolutional layers and flattened
    :return: The function `encoder` returns a Sequential model that consists of several Conv1D layers
    with increasing depth, followed by LeakyReLU activation, BatchNormalization, and Dropout layers.
    Finally, it includes a Flatten layer and a Dense layer with the specified `EncodeSize` and 'relu'
    activation function.
    """
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

# AutoEncoder
def autoEncoder(Encoder, Generator):
    """
    The function `autoEncoder` creates a model by stacking an Encoder and a Generator using the
    Sequential API in Keras.
    
    :param Encoder: The `Encoder` parameter in the `autoEncoder` function is expected to be a neural
    network model that encodes the input data into a lower-dimensional representation. This
    lower-dimensional representation is then passed to the `Generator` for decoding
    :param Generator: The `Generator` parameter in the `autoEncoder` function is typically a neural
    network model that takes encoded data from the encoder and generates output data that closely
    resembles the input data. In the context of autoencoders, the generator is responsible for
    reconstructing the original input data from the encoded representation
    :return: The function `autoEncoder` is returning a Keras model that consists of the Encoder and
    Generator layers stacked sequentially.
    """
    model = Sequential()
    model.add(Encoder)
    model.add(Generator)
    return model