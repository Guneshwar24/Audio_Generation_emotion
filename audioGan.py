import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
import IPython
import IPython.display as ipd 
import os
from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.models import load_model
from ganModels import *
from ganSetup import *
class AudioGAN:
    def __init__(self, label="AudioGAN", load=False, model_path="./models/"):
        print(label)
        self.config = GANConfig()
        self.model_path = model_path

        if load:
            self.load_models()
        else:
            self.initialize_models()
            self.trainData = normalization(load_train_data(self.config.AUDIO_SHAPE))

        self.disLossHist = []
        self.genLossHist = []

    def initialize_models(self):
        """ Initialize GAN models. """
        
        # self.gen = generator(self.config.NOISE_DIM, self.config.AUDIO_SHAPE)
        self.dis = discriminator(self.config.AUDIO_SHAPE)
        # self.enc = encoder(self.config.AUDIO_SHAPE, self.config.ENCODE_SIZE)
        # self.autoencoder = autoEncoder(self.enc, self.gen)

        self.enc = encoder(self.config.AUDIO_SHAPE, self.config.ENCODE_SIZE)
        self.gen = generator(self.config.NOISE_DIM, self.config.AUDIO_SHAPE)
        self.autoencoder = autoEncoder(self.enc, self.gen)

        self.autoencoder.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        self.gen.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        self.dis.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.stackGenDis = self.create_stacked_model(self.gen, self.dis)

    def train_autoencoder(self, train_data, epochs=20, batch_size=32):
        """ Train the autoencoder separately to minimize reconstruction error. """
        self.autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        # Initialize the Generator after Autoencoder is trained
        print("Autoencoder training complete.")
        # After training the autoencoder, initialize the GAN generator
        self.initialize_generator_from_encoder()

    def initialize_generator_from_encoder(self):
        # print("Starting weight transfer from encoder to generator.")
        # for enc_layer, gen_layer in zip(self.enc.layers, self.gen.layers):
        #     print(f"Checking {enc_layer.name} to {gen_layer.name}")
        #     if len(enc_layer.get_weights()) > 0 and len(gen_layer.get_weights()) > 0:
        #         if enc_layer.get_weights()[0].shape == gen_layer.get_weights()[0].shape:
        #             print(f"Transferring weights from {enc_layer.name} to {gen_layer.name}")
        #             gen_layer.set_weights(enc_layer.get_weights())
        #         else:
        #             print(f"Skipping weight transfer for {enc_layer.name} due to shape mismatch.")
        #     else:
        #         print(f"Skipping {enc_layer.name} and {gen_layer.name} as one or both do not have weights.")

        # Transfer weights from the encoder to the generator's corresponding layers
        encoder_weights = [layer.get_weights() for layer in self.enc.layers if len(layer.get_weights()) > 0]
        generator_layers = [layer for layer in self.gen.layers if len(layer.get_weights()) > 0]
        
        # Assuming a similar architecture for simplicity
        for e_weights, g_layer in zip(encoder_weights, generator_layers):
            if e_weights[0].shape == g_layer.get_weights()[0].shape:
                g_layer.set_weights(e_weights)
                print(f"Weights transferred to {g_layer.name}")
                
    
    def create_stacked_model(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential([generator, discriminator])
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return model

    def save_models(self):
        """ Save all models to files using the native Keras format and handle existing files. """
        # Define paths for each model, now using `.keras` extension
        generator_path = os.path.join(self.model_path, 'generator.keras')
        discriminator_path = os.path.join(self.model_path, 'discriminator.keras')
        autoencoder_path = os.path.join(self.model_path, 'autoencoder.keras')

        # Ensure the directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Check if files exist and remove them if they do
        for path in [generator_path, discriminator_path, autoencoder_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed existing model file at {path}")

        # Save each model using the new paths
        self.gen.save(generator_path)
        print(f"Generator model saved at {generator_path}")

        self.dis.save(discriminator_path)
        print(f"Discriminator model saved at {discriminator_path}")

        self.autoencoder.save(autoencoder_path)
        print(f"Autoencoder model saved at {autoencoder_path}")

        
    def load_models(self):
        """ Load all models from files using the TensorFlow SavedModel format. """
        # Define paths for each model
        generator_path = os.path.join(self.model_path, 'generator')
        discriminator_path = os.path.join(self.model_path, 'discriminator')
        autoencoder_path = os.path.join(self.model_path, 'autoencoder')

        # Load each model if the directory exists
        if os.path.exists(generator_path):
            self.gen = load_model(generator_path)
            print(f"Generator model loaded from {generator_path}")
        else:
            print("Generator model could not be found.")

        if os.path.exists(discriminator_path):
            self.dis = load_model(discriminator_path)
            print(f"Discriminator model loaded from {discriminator_path}")
        else:
            print("Discriminator model could not be found.")

        if os.path.exists(autoencoder_path):
            self.autoencoder = load_model(autoencoder_path)
            print(f"Autoencoder model loaded from {autoencoder_path}")
        else:
            print("Autoencoder model could not be found.")

        # Reinitialize stacked model as it depends on the loaded generator and discriminator
        self.stackGenDis = self.create_stacked_model(self.gen, self.dis)
        self.initialize_models()
            
    def train_gan(self, epochs=20, batch=32, save_interval=2):
        # half_batch = batch // 2
        # for epoch in range(epochs):
        #     # Training discriminator on real and generated data
        #     random_index = np.random.randint(0, len(self.trainData) - half_batch)
        #     legit_audios = self.trainData[random_index: random_index + half_batch]
        #     legit_audios = legit_audios.reshape((-1, self.config.AUDIO_SHAPE, 1))
        #     gen_noise = np.random.normal(0, 1, (half_batch, self.config.NOISE_DIM))
        #     synthetic_audios = self.gen.predict(gen_noise)
        #     x_combined_batch = np.concatenate((legit_audios, synthetic_audios))
        #     y_combined_batch = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))
        #     d_loss = self.dis.train_on_batch(x_combined_batch, y_combined_batch)

        #     # Append discriminator loss to history
        #     self.disLossHist.append(d_loss[0] if isinstance(d_loss, list) else d_loss)

        #     # Training generator via the stacked model
        #     noise = np.random.normal(0, 1, (batch, self.config.NOISE_DIM))
        #     y_mislabeled = np.ones((batch, 1))
        #     g_loss = self.stackGenDis.train_on_batch(noise, y_mislabeled)

        #     # Append generator loss to history
        #     self.genLossHist.append(g_loss[0] if isinstance(g_loss, list) else g_loss)

        #     # Logging the progress
        #     if epoch % save_interval == 0:
        #         self.log_progress(epoch, d_loss, g_loss)
        #         self.show_gen_samples(4)
        
        for epoch in range(epochs):
            # Sample noise as input for the generator
            noise = np.random.normal(0, 1, (batch, self.config.NOISE_DIM))
            generated_audios = self.gen.predict(noise)

            # Get real audio data samples
            idx = np.random.randint(0, self.trainData.shape[0], batch)
            real_audios = self.trainData[idx].reshape(-1, self.config.AUDIO_SHAPE, 1)

            # Train discriminator
            d_loss_real = self.dis.train_on_batch(real_audios, np.ones((batch, 1)))
            d_loss_fake = self.dis.train_on_batch(generated_audios, np.zeros((batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = self.stackGenDis.train_on_batch(noise, np.ones((batch, 1)))

            if epoch % save_interval == 0:
                print(f'Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}')


    def log_progress(self, epoch, d_loss, g_loss):
        # Handle both scalar and list outputs
        d_loss_val = d_loss[0] if isinstance(d_loss, list) else d_loss
        g_loss_val = g_loss[0] if isinstance(g_loss, list) else g_loss
        print(f"Epoch: {epoch}, Discriminator loss: {d_loss_val}, Generator loss: {g_loss_val}")

    def show_gen_samples(self, samples=4):
        noise = np.random.normal(0, 1, (samples, self.config.NOISE_DIM))
        audios = self.gen.predict(noise)
        audios = np.clip(audios, -1, 1)  # Ensure audio values are within [-1, 1]
        for i, audio in enumerate(audios):
            audio = audio.flatten() if audio.ndim > 1 else audio
            ipd.display(ipd.Audio(data=audio, rate=self.config.SAMPLE_RATE))

    # Train autoencoder
    # def train_autoencoder(self, epochs = 20,  save_internal = 2, batch = 32):
    #     config = GANConfig()
    #     for cnt in range(epochs):
    #         random_index = np.random.randint(0, len(self.trainData) - batch)
    #         legit_audios = self.trainData[random_index: int(random_index + batch)]

    #         # Ensure the audio data is in the correct shape (batch_size, AUDIO_SHAPE, 1)
    #         legit_audios = legit_audios.reshape((-1, config.AUDIO_SHAPE, 1))  # Reshape for 1D convolution
    #         loss = self.autoencoder.train_on_batch(legit_audios, legit_audios)
    #         # print("loss object",loss[0], loss[1])
    #         if cnt % save_internal == 0 : 
    #             print("Epoch: ", cnt, ", Loss: ", loss[0])