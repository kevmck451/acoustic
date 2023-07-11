# Functions to Process Audio

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from scipy import signal
import numpy as np
import librosa
import utils
import copy


# Function to convert audio sample to a specific length
def generate_chunks(audio_object, length):
    num_samples = audio_object.sample_rate * length
    start = 0
    end = num_samples
    total_samples = len(audio_object.data)

    audio_ob_list = []

    # If the audio file is too short, pad it with zeroes
    if total_samples < num_samples:
        audio_object.data = np.pad(audio_object.data, (0, num_samples - len(audio_object.data)))
        audio_ob_list.append(audio_object)
    # If the audio file is too long, shorten it

    elif total_samples > num_samples:
        while end <= total_samples:
            audio_copy = deepcopy(audio_object)
            audio_copy.data = audio_object.data[start:end]
            audio_ob_list.append(audio_copy)
            start, end = (start + num_samples), (end + num_samples)

    return audio_ob_list

# Function to convert 4 channel wav to list of 4 objects
def channel_to_objects(audio_object):
    audio_a = deepcopy(audio_object)
    audio_a.data = audio_object.data[0]
    audio_b = deepcopy(audio_object)
    audio_b.data = audio_object.data[1]
    audio_c = deepcopy(audio_object)
    audio_c.data = audio_object.data[2]
    audio_d = deepcopy(audio_object)
    audio_d.data = audio_object.data[3]

    return [audio_a, audio_b, audio_c, audio_d]

# Function to calculate MFCC of audio
def mfcc(audio_object, n_mfcc=50):
    # Extract the data from the audio object
    data = audio_object.data

    # Initialize an empty list to store the MFCCs for each channel
    mfccs_all_channels = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate MFCCs for this channel
        mfccs = librosa.feature.mfcc(y=channel_data, sr=audio_object.sample_rate, n_mfcc=n_mfcc)

        # Normalize the MFCCs
        mfccs = StandardScaler().fit_transform(mfccs)

        # Append to the list
        mfccs_all_channels.append(mfccs)

    # Convert the list of MFCCs to a numpy array and return
    return np.array(mfccs_all_channels)

# Function to calculate spectrogram of audio
def spectrogram(audio_object, range=(80, 2000), **kwargs):
    stats = kwargs.get('stats', False)
    window_size = 32768
    hop_length = 512
    frequency_range = range

    Audio_Object = normalize(audio_object)
    data = Audio_Object.data

    # Initialize an empty list to store the spectrograms for each channel
    spectrograms = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = np.abs(librosa.stft(channel_data, n_fft=window_size, hop_length=hop_length)) ** 2

        # Convert to decibels (log scale) for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Calculate frequency range and resolution
        nyquist_frequency = audio_object.sample_rate / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)
        frequency_range = np.arange(0, window_size // 2 + 1) * frequency_resolution

        bottom_index = int(np.round(range[0] / frequency_resolution))
        top_index = int(np.round(range[1] / frequency_resolution))

        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range: ({range[0]},{range[1]}) Hz')
            print(f'Freq Resolution: {frequency_resolution} Hz')

        # Cut the spectrogram to the desired frequency range and append to the list
        spectrograms.append(spectrogram_db[bottom_index:top_index])

    # Convert the list of spectrograms to a numpy array and return
    return np.array(spectrograms)

# Function to calculate the Signal to Noise Ratio with PSD
def signal_noise_ratio_psd(signal, noise, ):
    frequencies, psd_signal = power_spectral_density(signal)
    frequencies , psd_noise = power_spectral_density(noise)

    snr = 10 * np.log10(psd_signal / psd_noise)
    return frequencies, snr

# Function to calculate the Signal to Noise Ratio with RMS
def signal_noise_ratio_rms(signal, noise):
    rms_sig = root_mean_square(signal)
    rms_noise = root_mean_square(noise)
    snr = 10 * np.log10(rms_sig / rms_noise)
    return snr

# Function to calculate the Power Spectral Density
def power_spectral_density(audio_object):
    # freq is list of frequencies and psd is array with number of channels
    # that contains the power at those frequencies power (watts) per Hz

    frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate/4, nfft=32768) # 2048, 4096, 8192, 16384
    # frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate, average='mean') # 2048, 4096, 8192, 16384


    from matplotlib import pyplot as plt
    import pandas as pd

    avg_freq, avg_data = signal.welch(x=audio_object.data[0], fs=audio_object.sample_rate/4, average='mean')
    avg_data_pd = pd.Series(avg_data).rolling(13, center=True).mean().to_numpy()
    plt.figure(1, figsize=(14,8)).clf()
    plt.semilogy(avg_freq, avg_data, label='Raw PSD', lw=1, alpha=0.75)
    plt.semilogy(avg_freq, avg_data_pd, label='Rolled', lw=1, alpha=0.75)
    plt.title('\nPower Spectral Density Estimate')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power Spectral Density $\left(\frac{V^{2}}{Hz}\right)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return frequencies, psd

# Function to calculate the RMS Power
def root_mean_square(audio_object):
    rms = np.sqrt(np.mean(np.square(audio_object.data)))

    return rms

# Function to Increase or Decrease Sample Gain
def amplify(audio_object, gain_db):
    Audio_Object_amp = copy.deepcopy(audio_object)

    # convert gain from decibels to linear scale
    gain_linear = 10 ** (gain_db / 20)

    # multiply the audio data by the gain factor
    Audio_Object_amp.data *= gain_linear

    return Audio_Object_amp

# Function to Normalize Data
def normalize(audio_object, percentage=95):
    # make a deep copy of the audio object to preserve the original
    audio_normalized = copy.deepcopy(audio_object)
    max_value = np.max(np.abs(audio_normalized.data))
    normalized_data = audio_normalized.data / max_value * (percentage / 100.0)

    audio_normalized.data = normalized_data

    return audio_normalized



class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)