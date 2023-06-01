import librosa
import matplotlib.pyplot as plt
import numpy as np

directory = '/Volumes/KM1TB/Orlando Files/Audio/Samples/'
file_name = 'Hex_Flight_5.wav'

file_path = directory + file_name

CHANNEL_NUM = 4
CHANNEL_INDICES = [0, 1, 2, 3]
SAMPLE_RATE = 48000

# Load the audio file
audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
# audio: NumPy array containing the audio samples
# sr: Sampling rate of the audio file
print(audio.shape)

# Print the duration and sampling rate of the audio
duration = librosa.get_duration(y=audio, sr=sr)
print("Duration:", duration, "seconds")
print("Sampling Rate:", sr, "Hz")




# Check sample rate and change if necessary
# librosa.resample()

# Segmentation


# Feature Extraction


# Normalization


# Data Augmentation


# Data Splitting



