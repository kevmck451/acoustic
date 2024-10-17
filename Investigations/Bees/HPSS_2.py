import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# Load the audio file
y, sr = librosa.load('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/5 Bee Project/Data Collection/10-17-24/Synced/Bee Hive 1 chunk 1.wav', sr=None)

# Separate harmonic and percussive components
harmonic, percussive = librosa.effects.hpss(y)

# Design a low-pass filter to target percussive elements
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# Apply lowpass filter to percussive component
filtered_percussive = lowpass_filter(percussive, cutoff=1000, fs=sr)

# Subtract filtered percussive from the original signal
cleaned_audio = y - filtered_percussive

# Save the cleaned audio
sf.write('cleaned_audio.wav', cleaned_audio, sr)
