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
    def __init__(self, label="AudioGAN"):
        print(f"{label} initialized")
        
        # Audio Configurations
        self.sr = 16000  # Sample rate
        self.duration = 4  # Duration in seconds
        self.hop_length = 512  # Hop length for STFT
        self.n_mels = 128  # Number of Mel bands
        self.n_fft = 2048  # FFT window size
        self.noise_dim = 100  # Dimension of the noise vector for the generator
        
        self.input_shape = (self.n_mels, 1 + int(np.floor(self.sr * self.duration / self.hop_length)), 1)
        
        # Initialize models
        self.encoder = encoder(self.input_shape, 100)
        self.generator = generator(self.noise_dim, self.input_shape)
        self.discriminator = discriminator(self.input_shape)
        self.stackGenDis = stacked_G_D(self.generator, self.discriminator)
        self.autoencoder = autoEncoder(self.encoder, self.generator)
        
        # Set training data
        self.load_data()
        
        # Compile models
        self.compile_models()
    
    def compile_models(self):
        dis_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        auto_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        enc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer, metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
        self.autoencoder.compile(loss='mse', optimizer=auto_optimizer)
        self.stackGenDis.compile(loss='binary_crossentropy', optimizer=enc_optimizer, metrics=['accuracy'])

    def train_gan(self, epochs=50, batch_size=32):
        print("Starting GAN training...")
        for epoch in range(epochs):
            for _ in range(len(self.trainData) // batch_size):
                idx = np.random.randint(0, len(self.trainData) - batch_size)
                real_data = self.trainData[idx:idx + batch_size]

                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)

                real_labels = np.ones((batch_size, 1))
                fake_labels = np.zeros((batch_size, 1))

                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.stackGenDis.train_on_batch(noise, real_labels)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    def load_data(self):
        # Load training data from predefined functions
        raw_data = load_mel_spectrograms()
        self.trainData = normalization(raw_data)
        print(f"Training data loaded and normalized, shape: {self.trainData.shape}")


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
    def train_autoencoder(self, epochs=10, save_internal=2, batch_size=32):
        print("Starting autoencoder training...")
        for epoch in range(epochs):
            for _ in range(len(self.trainData) // batch_size):
                idx = np.random.randint(0, len(self.trainData) - batch_size)
                legit_audios = self.trainData[idx:idx + batch_size]

                print("Shape before reshaping:", legit_audios.shape)
                print("Expected elements:", np.prod(legit_audios.shape))
                print("Target shape elements:", np.prod((batch_size,) + AUDIO_SHAPE))

                try:
                    legit_audios = legit_audios.reshape((batch_size,) + AUDIO_SHAPE)
                    loss = self.autoencoder.train_on_batch(legit_audios, legit_audios)
                    if (epoch + 1) % save_internal == 0:
                        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss}")
                except ValueError as e:
                    print(f"Reshape error: {e}")
