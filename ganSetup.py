import librosa
import numpy as np
import pandas as pd
import os
import soundfile as sf
class GANConfig:
    DURATION = 4
    SAMPLE_RATE = 16000
    AUDIO_SHAPE = SAMPLE_RATE * DURATION
    NOISE_DIM = 500
    MFCC = 40
    ENCODE_SIZE = NOISE_DIM
    DENSE_SIZE = 2100
    DATASET_PATH = "./Datasets/mosei/"
    AUTO_ENCODER_PATH = "./WavFiles/Autoencoder/"
    PICTURE_PATH = "./Pictures/"
    GAN_PATH = "./WavFiles/GAN/"
    LABEL = "fname"

def load_train_data(input_length):
    train = pd.read_csv(GANConfig.DATASET_PATH + "mosei_train_updated.csv")
    X = np.empty((len(train), input_length))
    for i, fname in enumerate(train['fname']):
        file_path = GANConfig.DATASET_PATH + "audio_train/" + fname
        try:
            data, _ = librosa.load(file_path, sr=GANConfig.SAMPLE_RATE, res_type='kaiser_fast')
            if len(data) > input_length:
                offset = np.random.randint(len(data) - input_length)
                data = data[offset:offset + input_length]
            else:
                data = np.pad(data, (0, input_length - len(data)), 'constant')
            X[i] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            X[i] = np.zeros(input_length)
    return X

def normalization(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / std

def generate_and_save_audio(generator, test_data, output_path, sample_rate):
    """Generate audio samples using the GAN generator and save them to disk."""
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate audio samples
    generated_samples = generator.predict(test_data)
    
    # Save the generated audio files
    for i, sample in enumerate(generated_samples):
        # Reshape the sample if necessary
        sample = np.squeeze(sample)
        file_path = os.path.join(output_path, f"generated_sample_{i}.wav")
        sf.write(file_path, sample, samplerate=sample_rate)
        print(f"Generated audio saved to {file_path}")