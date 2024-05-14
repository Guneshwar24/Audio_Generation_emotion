import librosa
import numpy as np
import pandas as pd
import os
import soundfile as sf

# The `GANConfig` class defines configuration parameters for a GAN (Generative Adversarial Network)
# model.
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
    """
    The function `load_train_data` reads audio files from a CSV, processes them using librosa, and
    returns a numpy array of the audio data.
    
    :param input_length: The `input_length` parameter in the `load_train_data` function represents the
    desired length of the audio data samples that will be loaded from the files. This parameter
    specifies the length to which each audio sample will be either trimmed or zero-padded to ensure
    consistency in the input data for further processing or
    :return: The function `load_train_data` returns a NumPy array `X` containing audio data loaded from
    files specified in the "mosei_train_filtered.csv" dataset. The audio data is processed to ensure it
    has a specified input length, either by truncating or padding with zeros as needed. If an error
    occurs while loading a file, the function prints an error message and fills the corresponding entry
    in
    """
    train = pd.read_csv("mosei_train_filtered.csv")
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

def load_test_data(input_length):
    """
    The function `load_test_data` reads test data from a CSV file, loads audio files, and preprocesses
    them to a specified input length.
    
    :param input_length: The `input_length` parameter in the `load_test_data` function represents the
    desired length of the audio data samples that will be loaded from the test files. This parameter is
    used to ensure that all audio samples have a consistent length for processing in the machine
    learning model
    :return: The function `load_test_data` returns a NumPy array `X` containing audio data from test
    files, with each row representing the audio data for a specific file.
    """
    test = pd.read_csv("mosei_test_filtered.csv")
    print("length of test files", len(test))
    X = np.empty((len(test), input_length))
    for i, fname in enumerate(test['fname']):
        file_path = GANConfig.DATASET_PATH + "audio_test/" + fname
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
    """
    This function generates audio samples using a GAN generator and saves them to disk as WAV files.
    
    :param generator: The `generator` parameter in the `generate_and_save_audio` function is typically a
    Generative Adversarial Network (GAN) model that has been trained to generate audio samples. This
    model takes random input data (often referred to as latent space vectors) as input and generates
    audio samples as output
    :param test_data: Test data to generate audio samples from
    :param output_path: The `output_path` parameter in the `generate_and_save_audio` function is the
    directory path where the generated audio samples will be saved to disk. It is the location where the
    function will save the audio files in WAV format
    :param sample_rate: The `sample_rate` parameter in the `generate_and_save_audio` function refers to
    the number of samples per second of audio. It is a measure of the audio quality and is typically
    expressed in Hertz (Hz). Common sample rates for audio files are 16000 Hz and
    """
    """Generate audio samples using the GAN generator and save them to disk."""
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate audio samples
    generated_samples = generator.predict(test_data)
    
    # Save the generated audio files
    for i, sample in enumerate(generated_samples):
        sample = np.squeeze(sample)
        file_path = os.path.join(output_path, f"generated_sample_{i}.wav")
        sf.write(file_path, sample, samplerate=sample_rate)
        print(f"Generated audio saved to {file_path}")