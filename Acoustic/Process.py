# Functions to Process Audio

import numpy as np


class Process:
    def __init__(self, Audio_Object):
        self.Audio_Object = Audio_Object



# Function to Increase or Decrease Sample Gain
def amplify(Audio_Object):
    return Audio_Object


# Function to Normalize Data
def normalize(Audio_Object):
    max_value = np.max(np.abs(Audio_Object.data))  # Maximum absolute value
    normalized_data = Audio_Object.data / max_value

    Audio_Object.data = normalized_data

    return Audio_Object