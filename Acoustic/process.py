# Functions to Process Audio

import numpy as np
import copy
import utils
from scipy import signal


class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)



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

