# Functions to Process Audio

import numpy as np

# Function to Normalize Data
def normalize(Audio_Object):
    max_value = np.max(np.abs(Audio_Object.data))  # Maximum absolute value
    normalized_data = Audio_Object.data / max_value

    Audio_Object.data = normalized_data

    return Audio_Object