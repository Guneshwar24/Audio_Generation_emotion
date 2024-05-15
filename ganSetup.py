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
    The function `load_train_data` reads audio files, computes MFCC features, and prepares the data for
    training by adjusting the frame length and handling errors with zero padding.
    
    :param input_length: The `input_length` parameter in the `load_train_data` function represents the
    number of frames you want to use for each audio sample. This parameter determines the length of the
    input data that will be processed for each audio file
    :return: The function `load_train_data(input_length)` returns a NumPy array `X` containing MFCC
    features for audio files in the training dataset. The shape of the array is `(number of samples,
    input_length, 20)`, where `input_length` is the number of frames specified as input. Each sample in
    the array represents MFCC features for an audio file, with each frame containing
    """
    train = pd.read_csv("mosei_train_filtered.csv")
    X = np.empty((len(train), input_length, 20))  # Assuming 20 MFCCs, and input_length is the number of frames
    for i, fname in enumerate(train['fname']):
        file_path = GANConfig.DATASET_PATH + "audio_train/" + fname
        try:
            # Load and compute MFCC features
            data, _ = librosa.load(file_path, sr=GANConfig.SAMPLE_RATE, res_type='kaiser_fast')
            # data = librosa.feature.mfcc(y=librosa.load(file_path, sr=GANConfig.SAMPLE_RATE, mono=True, res_type='kaiser_fast')[0], sr=GANConfig.SAMPLE_RATE, n_mfcc=20, n_fft=int(GANConfig.SAMPLE_RATE*0.025), hop_length=int(GANConfig.SAMPLE_RATE*0.01))
            # Ensure the MFCC output matches the desired frame length
            if data.shape[1] > input_length:
                offset = np.random.randint(data.shape[1] - input_length)
                data = data[:, offset:offset + input_length]
            else:
                data = np.pad(data, ((0, 0), (0, input_length - data.shape[1])), 'constant')

            X[i] = data.T  # Transpose to match the shape (64000, 20)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            X[i] = np.zeros((input_length, 20))  # Ensure zero padding matches (64000, 20) if there's an error
    return X

def load_test_data(input_length):
    """
    The function `load_test_data` reads audio files from a CSV file, extracts MFCC features using
    librosa, and pads or truncates the features to match the specified input length.
    
    :param input_length: The `input_length` parameter in the `load_test_data` function represents the
    desired length of the input data for each sample. This function reads test data from a CSV file,
    loads audio files, extracts MFCC features, and prepares the input data for a machine learning model.
    The input data for
    :return: The function `load_test_data(input_length)` returns a NumPy array `X` containing MFCC
    features extracted from audio files in the "mosei_test_filtered.csv" dataset. The shape of the array
    is `(len(test), input_length)`, where `len(test)` is the number of audio files in the test dataset
    and `input_length` is the specified length for the MFCC features
    """
    test = pd.read_csv("mosei_test_filtered.csv")
    print("length of test files", len(test))
    X = np.empty((len(test), input_length))
    for i, fname in enumerate(test['fname']):
        file_path = GANConfig.DATASET_PATH + "audio_test/" + fname
        try:
            data, _ = librosa.load(file_path, sr=GANConfig.SAMPLE_RATE, res_type='kaiser_fast')
            # data = librosa.feature.mfcc(y=librosa.load(file_path, sr=GANConfig.SAMPLE_RATE, mono=True, res_type='kaiser_fast')[0], sr=GANConfig.SAMPLE_RATE, n_mfcc=20, n_fft=int(GANConfig.SAMPLE_RATE*0.025), hop_length=int(GANConfig.SAMPLE_RATE*0.01))
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