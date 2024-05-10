import numpy as numpy
import IPython.display as ipd 
import os
from keras.optimizers import Adam
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
            print("Loaded")
        else:
            self.initialize_models()
        
        self.trainData = normalization(load_train_data(self.config.AUDIO_SHAPE))
        self.testData = normalization(load_test_data(self.config.AUDIO_SHAPE))

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
        generator_path = os.path.join(self.model_path, 'generator.keras')
        discriminator_path = os.path.join(self.model_path, 'discriminator.keras')
        autoencoder_path = os.path.join(self.model_path, 'autoencoder.keras')

        models_loaded = True

        # Load each model if the directory exists
        if os.path.exists(generator_path):
            self.gen = load_model(generator_path)
            print(f"Generator model loaded from {generator_path}")
        else:
            print("Generator model could not be found.")
            models_loaded = False

        if os.path.exists(discriminator_path):
            self.dis = load_model(discriminator_path)
            print(f"Discriminator model loaded from {discriminator_path}")
        else:
            print("Discriminator model could not be found.")
            models_loaded = False

        if os.path.exists(autoencoder_path):
            self.autoencoder = load_model(autoencoder_path)
            print(f"Autoencoder model loaded from {autoencoder_path}")
        else:
            print("Autoencoder model could not be found.")
            models_loaded = False

        # Only create the stacked model if all components are loaded successfully
        if models_loaded:
            self.stackGenDis = self.create_stacked_model(self.gen, self.dis)
        else:
            # Initialize models if any of the required models could not be loaded
            self.initialize_models()

        
    def log_progress(self, epoch, d_loss, g_loss):
        # Handle both scalar and list outputs
        d_loss_val = d_loss[0] if isinstance(d_loss, list) else d_loss
        g_loss_val = g_loss[0] if isinstance(g_loss, list) else g_loss
        print(f"Epoch: {epoch}, Discriminator loss: {d_loss_val}, Generator loss: {g_loss_val}")
            
    def train_gan(self, epochs=20, batch=32, save_interval=2):
        for epoch in range(epochs):
            for _ in range(int(self.trainData.shape[0] / batch)):
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch, self.config.NOISE_DIM))

                # Generate a batch of new audio samples
                generated_audios = self.gen.predict(noise, verbose=0)

                # Get a random batch of real audio samples
                idx = np.random.randint(0, self.trainData.shape[0], batch)
                real_audios = self.trainData[idx].reshape(-1, self.config.AUDIO_SHAPE, 1)

                # Train the discriminator (real classified as ones and fake as zeros)
                d_loss_real = self.dis.train_on_batch(real_audios, np.ones((batch, 1)))
                d_loss_fake = self.dis.train_on_batch(generated_audios, np.zeros((batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator (trick the discriminator to classify fakes as real)
                misleading_targets = np.ones((batch, 1))
                g_loss = self.stackGenDis.train_on_batch(noise, misleading_targets)

                # Optionally plot the progress
                if epoch % save_interval == 0:
                    d_loss_val = d_loss[0] if isinstance(d_loss, list) else d_loss
                    g_loss_val = g_loss[0] if isinstance(g_loss, list) else g_loss
                    #Appending to the variables
                    self.disLossHist.append(d_loss_val)
                    self.genLossHist.append(g_loss)
                    disLossHist_first = [loss[1] for loss in self.disLossHist]
                    print(f"Epoch: {epoch}, Discriminator loss: {disLossHist_first}, Generator loss: {g_loss_val}")
         


    def show_gen_samples(self, samples=4):
        noise = np.random.normal(0, 1, (samples, self.config.NOISE_DIM))
        audios = self.gen.predict(noise)
        audios = np.clip(audios, -1, 1)  # Ensure audio values are within [-1, 1]
        for i, audio in enumerate(audios):
            audio = audio.flatten() if audio.ndim > 1 else audio
            ipd.display(ipd.Audio(data=audio, rate=self.config.SAMPLE_RATE))

    # Train autoencoder
    def train_autoencoder(self, epochs = 20,  save_internal = 2, batch = 32):
        config = GANConfig()
        for cnt in range(epochs):
            random_index = np.random.randint(0, len(self.trainData) - batch)
            legit_audios = self.trainData[random_index: int(random_index + batch)]
            test_audios = self.testData[random_index: int(random_index + batch)]

            # Ensure the audio data is in the correct shape (batch_size, AUDIO_SHAPE, 1)
            legit_audios = legit_audios.reshape((-1, config.AUDIO_SHAPE, 1))  # Reshape for 1D convolution
            loss = self.autoencoder.fit(legit_audios, legit_audios, epochs, batch, shuffle=True, validation_data=(test_audios, test_audios))

            if cnt % save_internal == 0 : 
                print("Epoch: ", cnt, ", Loss: ", loss[0])