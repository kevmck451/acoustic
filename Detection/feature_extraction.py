

from Acoustic.audio import Audio


import numpy as np



def extract_features(path, duration):
    num_samples = 48000 * duration
    # Load audio file with fixed sample rate
    audio = Audio(path)

    # If the audio file is too short, pad it with zeroes
    if len(audio.data) < num_samples:
        audio.data = np.pad(audio.data, (0, num_samples - len(audio.data)))
    # If the audio file is too long, shorten it
    elif len(audio.data) > num_samples:
        audio.data = audio.data[:num_samples]

    # Feature Extraction
    # mfccs = audio.mfcc()
    spectro = audio.spectrogram()

    # return mfccs
    return spectro