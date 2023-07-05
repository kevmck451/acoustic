

from Acoustic.audio import Audio


from pathlib import Path
import numpy as np



# Load Data from a Dataset with Labels

def load_audio_data(path, label, duration=10):
    X = []
    y = []

    # Calculate the number of samples in the audio file
    num_samples = 48000 * duration

    for file in Path(path).rglob('*.wav'):
        # Load audio file with fixed sample rate
        audio = Audio(str(file))

        # If the audio file is too short, pad it with zeroes
        if len(audio.data) < num_samples:
            audio.data = np.pad(audio.data, (0, num_samples - len(audio.data)))
            print('too short')

        # If the audio file is too long, truncate it
        elif len(audio.data) > num_samples:
            audio.data = audio.data[:num_samples]
            print('too long')
            print(audio.filepath.stem)

        # Feature Extraction
        mfccs = audio.mfcc()

        X.append(mfccs)
        y.append(label)

    return X, y


# Path to audio samples
path_with_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/1')
path_without_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/0')

# -------- Load and preprocess data
X_with_signals, y_with_signals = load_audio_data(path_with_signals, 1)
X_without_signals, y_without_signals = load_audio_data(path_without_signals, 0)

X = X_with_signals + X_without_signals
y = y_with_signals + y_without_signals



# Select and Extract Feature to use for training







# Create a Model for Trainning







# Test accuracy of Model






# Save Model if Desired










