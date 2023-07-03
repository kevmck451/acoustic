# Train ML Model on a Dataset and save the model for testing


from pathlib import Path
import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


def load_audio_data(path, label, sample_rate=48000, duration=10, n_mfcc=13):
    X = []
    y = []

    # Calculate the number of samples in the audio file
    num_samples = sample_rate * duration

    for audio_file_path in Path(path).rglob('*.wav'):
        # Load audio file with fixed sample rate
        y_, sr = librosa.load(str(audio_file_path), sr=sample_rate)

        # If the audio file is too short, pad it with zeroes
        if len(y_) < num_samples:
            y_ = np.pad(y_, (0, num_samples - len(y_)))

        # If the audio file is too long, truncate it
        elif len(y_) > num_samples:
            y_ = y_[:num_samples]

        # Generate a fixed number of MFCCs
        mfccs = librosa.feature.mfcc(y=y_, sr=sample_rate, n_mfcc=n_mfcc)

        # Normalize mfccs
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

        X.append(mfccs)
        y.append(label)

    return X, y


def preprocess_data(X, y):
    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]  # 4D array for CNN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Path to audio samples
path_with_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static/1')
path_without_signals = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static/0')

# Load and preprocess data
X_with_signals, y_with_signals = load_audio_data(path_with_signals, 1)
X_without_signals, y_without_signals = load_audio_data(path_without_signals, 0)

X = X_with_signals + X_without_signals
y = y_with_signals + y_without_signals

X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Create and train model
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = create_model(input_shape)
model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))

saveto = 'models/audio_detection_model_test_0.h5'
num = 1
while Path(saveto).exists():
    saveto = f'models/audio_detection_model_test_{num}.h5'
    num += 1

# Save the model
model.save(saveto)










