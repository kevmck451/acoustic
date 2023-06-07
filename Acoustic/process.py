# Functions to Process Audio

import numpy as np
import copy
import utils
import sample_library


class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)








# Function to Increase or Decrease Sample Gain
def amplify(Audio_Object, gain_db):
    Audio_Object_amp = copy.deepcopy(Audio_Object)

    # convert gain from decibels to linear scale
    gain_linear = 10 ** (gain_db / 20)

    # multiply the audio data by the gain factor
    Audio_Object_amp.data *= gain_linear

    return Audio_Object_amp



# Function to Normalize Data
def normalize(Audio_Object, percentage=95):
    # make a deep copy of the audio object to preserve the original
    audio_normalized = copy.deepcopy(Audio_Object)
    max_value = np.max(np.abs(audio_normalized.data))
    normalized_data = audio_normalized.data / max_value * (percentage / 100.0)

    audio_normalized.data = normalized_data

    return audio_normalized