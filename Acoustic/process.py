# Functions to Process Audio

import numpy as np
import copy
import utils
from scipy import signal
import sample_library



class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)



# Function to calculate the Signal to Noise Ratio
def signal_noise_ratio(signal, noise, type):
    if type == 'PSD':
        frequencies, psd_signal = power_spectral_density(signal)
        frequencies , psd_noise = power_spectral_density(noise)

        snr = 10 * np.log10(psd_signal / psd_noise)
        return frequencies, snr

    else: pass

# Function to calculate the Power Spectral Density
def power_spectral_density(audio_object):
    # freq is list of frequencies and psd is array with number of channels
    # that contains the power at those frequencies power (watts) per Hz

    # frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate/4, nfft=32768)
    frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate/4, nfft=16384) # 2048, 4096, 8192, 16384

    return frequencies, psd

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

