import librosa
import numpy as np
import pandas as pd

# Audio Config
DURATION = 4
SAMPLE_RATE = 16000
HOP_LENGTH = 512  # samples
# AUDIO_SHAPE = SAMPLE_RATE*DURATION
# Calculate TIME_STEPS
TIME_STEPS = 1 + (DURATION * SAMPLE_RATE) // HOP_LENGTH
TIME_STEPS
AUDIO_SHAPE = (128, TIME_STEPS, 1)
# AUDIO_SHAPE = (128, 400, 1)

NOISE_DIM = 500
MFCC = 40

ENCODE_SIZE = NOISE_DIM
DENSE_SIZE = 2100

# Paths
DATASET_PATH      = "./Datasets/mosei/"
AUTO_ENCODER_PATH = "./WavFiles/Autoencoder/"
PICTURE_PATH      = "./Pictures/"
GAN_PATH          = "./WavFiles/GAN/"

#Label
LABEL = "fname"

# Load 
def load_mel_spectrograms():
    """
    Load audio files and convert each to a Mel spectrogram.
    """
    train = pd.read_csv(DATASET_PATH + "mosei_train_updated.csv")
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    max_pad_len = 400

    # Prepare the output array
    X = np.empty((len(train), n_mels, max_pad_len, 1))

    # Process each file
    for i, row in enumerate(train.itertuples()):
        file_path = DATASET_PATH + "audio_train/" + row.fname
        try:
            wave, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            wave = librosa.util.normalize(wave)
            mel_spectrogram = librosa.feature.melspectrogram(wave, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

            pad_width = max_pad_len - mel_spectrogram.shape[1]
            if pad_width < 0:
                mel_spectrogram = mel_spectrogram[:, :max_pad_len]
            else:
                mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')

            X[i, :, :, 0] = mel_spectrogram
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            X[i, :, :, 0] = np.zeros((n_mels, max_pad_len))  # Default to zeros on error

    return X

# def load_train_data(input_length=AUDIO_SHAPE):
#     # Load the dataset
#     train = pd.read_csv(DATASET_PATH + "mosei_train_updated.csv")

#     # Get the number of samples and initialize an array for storing audio data
#     cur_batch_size = len(train)
#     X = np.empty((cur_batch_size, input_length))

#     # Iterate through each filename in the dataframe
#     for i, train_fname in enumerate(train['fname']):
#         file_path = DATASET_PATH + "audio_train/" + train_fname

#         # Read and resample the audio
#         data, _ = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')

#         # Handle the audio length (either trim or pad)
#         if len(data) > input_length:
#             max_offset = len(data) - input_length
#             offset = np.random.randint(max_offset)
#             data = data[offset:(input_length+offset)]
#         else:
#             if input_length > len(data):
#                 max_offset = input_length - len(data)
#                 offset = np.random.randint(max_offset)
#             else:
#                 offset = 0
#             data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

#         # Store the processed audio data
#         X[i,] = data

#     print("Data Loaded...")
#     return X


# Stardize Data 
def normalization(X):
    mean = X.mean(keepdims=True)
    std = X.std(keepdims=True)
    X = (X - mean) / std
    print("Data Normalized...")
    return X

# Rescale Data to be in range [rangeMin, rangeMax]
def rescale(X, rangeMin=-1, rangeMax=+1):
    maxi = X.max()
    mini = X.min()
    X = np.interp(X, (mini, maxi), (rangeMin, rangeMax))
    print("Data Rescaled...")
    return X

