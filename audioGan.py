import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
import IPython
import IPython.display as ipd 

from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from ganModels import *
from ganSetup import *
class AudioGAN:
    def __init__(self, label="label"):
        print(label, "audioGan.py")
        # Generate models
        self.enc = encoder(AUDIO_SHAPE, ENCODE_SIZE)
        self.gen = generator(NOISE_DIM, AUDIO_SHAPE)
        self.dis = discriminator(AUDIO_SHAPE)
        self.stackGenDis = stacked_G_D(self.gen, self.dis)
        self.autoencoder = autoEncoder(self.enc, self.gen)  

        # Compile models with their own optimizer instances
        gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.9)
        dis_optimizer = Adam(learning_rate=0.0002, beta_1=0.9)
        ae_optimizer = Adam(learning_rate=0.0002, beta_1=0.9)
        stacked_optimizer = Adam(learning_rate=0.0002, beta_1=0.9)

        # Set different learning rates for the generator and discriminator
        gen_learning_rate = 0.00002  # Example value from the paper
        dis_learning_rate = 0.00002  # Example value from the paper

        gen_optimizer = Adam(learning_rate=gen_learning_rate, beta_1=0.5, beta_2=0.99)
        dis_optimizer = Adam(learning_rate=dis_learning_rate, beta_1=0.5, beta_2=0.99)

        self.gen.compile(loss='binary_crossentropy', optimizer=gen_optimizer, metrics=['accuracy'])
        self.dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer, metrics=['accuracy'])

        # self.gen.compile(loss='binary_crossentropy', optimizer=gen_optimizer, metrics=['accuracy'])
        # self.dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer, metrics=['accuracy'])
        self.autoencoder.compile(loss='mse', optimizer=ae_optimizer, metrics=['accuracy'])
        self.stackGenDis.compile(loss='binary_crossentropy', optimizer=stacked_optimizer, metrics=['accuracy'])

        # Set training data
        self.trainData = normalization(load_train_data(AUDIO_SHAPE))

        self.disLossHist = []
        self.genLossHist = []

    # Train GAN
    def train_gan(self, epochs = 20, batch = 32, save_interval = 2):
        for cnt in range(epochs):
          
            # Train discriminator
            halfBatch = int(batch/2)
            random_index = np.random.randint(0, len(self.trainData) - halfBatch)
            legit_audios = self.trainData[random_index: int(random_index + halfBatch)]
            
            # Reshape legit_audios to match the (batch_size, AUDIO_SHAPE, 1) format
            legit_audios = legit_audios.reshape((-1, AUDIO_SHAPE, 1))
            
            gen_noise = np.random.normal(0, 1, (halfBatch, NOISE_DIM))
            syntetic_audios = self.gen.predict(gen_noise)
            # Combine real and generated audios for discriminator training
            # x_combined_batch = np.concatenate((legit_audios, syntetic_audios))
            # y_combined_batch = np.concatenate((np.ones((halfBatch, 1)), np.zeros((halfBatch, 1))))
            x_combined_batch = np.concatenate((legit_audios, syntetic_audios), axis=0)
            y_combined_batch = np.concatenate((np.ones((halfBatch, 1)), np.zeros((halfBatch, 1))), axis=0)
            # Train discriminator on combined batch
            d_loss = self.dis.train_on_batch(x_combined_batch, y_combined_batch)
            
            # Update stacked discriminator weights
            self.stackGenDis.layers[1].set_weights(self.dis.get_weights())
    
            # Include discriminator loss
            d_loss_mean = np.mean(d_loss)
            self.disLossHist.append(d_loss_mean)
            # Train stacked generator
            noise = np.random.normal(0, 1, (batch, NOISE_DIM))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stackGenDis.train_on_batch(noise, y_mislabled)
            # Update generator Weights
            self.gen.set_weights(self.stackGenDis.layers[0].get_weights())
            
            # Include generator loss
            g_loss_mean = np.mean(g_loss)
            self.genLossHist.append(g_loss_mean)
            
            if cnt % int(save_interval/2) == 0:
                print("epoch: %d" % (cnt))
                print("Discriminator_loss: %f, Generator_loss: %f" % (d_loss_mean, g_loss_mean))
            if cnt % save_interval == 0:
                self.show_gen_samples(4)
     
    # Plot a number of generated samples
    def show_gen_samples(self, samples = 4):
        samplePlot = []
        fig        = plt.figure(figsize = (1, samples))
        noise      = np.random.normal(0, 1, (samples, self.gen.input_shape[1]))
        audios     = self.gen.predict(noise) 
        # Normalize or scale the audio to be between -1 and 1 if not already
        audios = np.clip(audios, -1, 1)  # Assuming audios are in floating point format       
        for i, audio in enumerate(audios):
            if len(audio.shape) > 1 and audio.shape[1] == 1:
                audio = audio.flatten()  # Flatten to 1D array if it's shaped as (length, 1)
            IPython.display.display(ipd.Audio(data = audio, rate = SAMPLE_RATE))
            samplePlot.append(fig.add_subplot(1, samples, i+1))
            samplePlot[i].plot(audio.flatten(), '.', )
        plt.gcf().set_size_inches(25, 5)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.show()

    # Train autoencoder
    def train_autoencoder(self, epochs = 20,  save_internal = 2, batch = 32):
        for cnt in range(epochs):
            random_index = np.random.randint(0, len(self.trainData) - batch)
            legit_audios = self.trainData[random_index: int(random_index + batch)]

            # Ensure the audio data is in the correct shape (batch_size, AUDIO_SHAPE, 1)
            legit_audios = legit_audios.reshape((-1, AUDIO_SHAPE, 1))  # Reshape for 1D convolution
            loss = self.autoencoder.train_on_batch(legit_audios, legit_audios)
            # print("loss object",loss[0], loss[1])
            if cnt % save_internal == 0 : 
                print("Epoch: ", cnt, ", Loss: ", loss[0])