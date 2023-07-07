

from Acoustic.audio import Audio
from Detection.test_model_accuracy import test_model_accuracy



from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from pathlib import Path
import numpy as np
import librosa
import process


# Inherit from Audio to preserve feature extraction settings
class Audio_Template(Audio):
    def __init__(self, path):
        super().__init__(path)

    # Function to calculate spectrogram of audio
    def spectrogram(self, range=(80, 2000), stats=False):
        # Do not change settings - ML Model depends on it as currently set
        window_size = 32768
        hop_length = 512
        frequency_range = range

        Audio_Object = process.normalize(self)
        data = Audio_Object.data

        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = np.abs(librosa.stft(data, n_fft=window_size, hop_length=hop_length)) ** 2

        # Convert to decibels (log scale) for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Calculate frequency range and resolution
        nyquist_frequency = self.SAMPLE_RATE / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)
        frequency_range = np.arange(0, window_size // 2 + 1) * frequency_resolution
        self.freq_range_low = int(frequency_range[0])
        self.freq_range_high = int(frequency_range[-1])
        self.freq_resolution = round(frequency_resolution, 2)

        bottom_index = int(np.round(range[0] / frequency_resolution))
        top_index = int(np.round(range[1] / frequency_resolution))

        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range: ({range[0]},{range[1]}) Hz')
            print(f'Freq Resolution: {self.freq_resolution} Hz')

        return spectrogram_db[bottom_index:top_index]

    # Function to calculate MFCC of audio
    def mfcc(self, n_mfcc=13):
        # Generate a fixed number of MFCCs
        mfccs = librosa.feature.mfcc(y=self.data, sr=self.SAMPLE_RATE, n_mfcc=n_mfcc)

        # Normalize mfccs
        mfccs = StandardScaler().fit_transform(mfccs)

        return mfccs

# Feature Extraction
def extract_features(path, duration):
    num_samples = 48000 * duration
    # Load audio file with fixed sample rate
    audio = Audio_Template(path)

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

# Load Data from a Dataset with Labels and Extract Features
def load_audio_data(path, duration=10):
    print('Loading Dataset')
    X = []
    y = []

    for file in Path(path).rglob('*.wav'):
        feature = extract_features(file, duration)
        X.append(feature) # Add Feature
        label = int(file.parent.stem)
        y.append(label) # Add Label (folder name)

    X = np.array(X)
    X = X[..., np.newaxis]
    return X, np.array(y)

#---------------------------------------------------------------
# Path to audio samples
static_dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/dataset')

# -------- Load and preprocess data
X, y = load_audio_data(static_dataset)

# Create a Model for Trainning
print('Creating Model')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1], X_train.shape[2], 1)
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
model.fit(X_train, y_train, epochs=20, batch_size=12, validation_data=(X_test, y_test))


# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Flatten())
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))
#
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#
# model.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Test accuracy of Model
directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/Test'
truth = {
    '10m-D-DEIdle_b': 1,
    '10m-D-TIdle_1_c': 1,
    'Hex_8_Hover_4_a': 0,
    'Hex_8_Hover_1_a': 0,
    '10m-D-TIdle_2_c': 1,
    'Hex_1_Takeoff_a': 0,
    '30m-D-DEIdle_a': 1,
    '30m-D-DEIdle_b': 1,
    '30m-D-DEIdle_c': 1,
    '30m-D-DEIdle_d': 1,
    '30m-D-TIdle_1_a': 1,
    '30m-D-TIdle_1_b': 1,
    '30m-D-TIdle_1_c': 1,
    '30m-D-TIdle_1_d': 1,
    '30m-D-TIdle_2_a': 1,
    '30m-D-TIdle_2_b': 1,
    '30m-D-TIdle_2_c': 1,
    '30m-D-TIdle_2_d': 1,
    '40m-D-DEIdle_a': 1,
    '40m-D-DEIdle_b': 1,
    '40m-D-DEIdle_c': 1,
    '40m-D-DEIdle_d': 1,
    '30m-D-Rev_a': 1,
    '30m-D-Rev_b': 1,
    '30m-D-Rev_c': 1,
    '30m-D-Rev_d': 1,
    '40m-D-Rev_a': 1,
    '40m-D-Rev_b': 1,
    '40m-D-Rev_c': 1,
    '40m-D-Rev_d': 1,
    '40m-D-TIdle_1_a': 1,
    '40m-D-TIdle_1_b': 1,
    '40m-D-TIdle_1_c': 1,
    '40m-D-TIdle_1_d': 1,
    '40m-D-TIdle_2_a': 1,
    '40m-D-TIdle_2_b': 1,
    '40m-D-TIdle_2_c': 1,
    '40m-D-TIdle_2_d': 1,
    'Hex_6_Flight1_a': 0,
    'Hex_6_Flight2_a': 0,
    'Hex_8_Hover_2_b': 0,
    'Hex_8_Hover_3_c': 0,
    'Hex_Hover_1_a': 0,
    'Hex_Hover_1_b': 0,
    'Hex_Hover_1_c': 0,
    'Hex_Hover_1_d': 0,
    'Hex_Hover_1b_a': 0,
    'Hex_Hover_1b_b': 0,
    'Hex_Hover_1b_c': 0,
    'Hex_Hover_1b_d': 0,
    'Hex_Hover_2_a': 0,
    'Hex_Hover_2_b': 0,
    'Hex_Hover_2_c': 0,
    'Hex_Hover_2_d': 0,
    'Hex_Hover_2b_a': 0,
    'Hex_Hover_2b_b': 0,
    'Hex_Hover_2b_c': 0,
    'Hex_Hover_2b_d': 0,
    'Hex_6_Hover_a': 0
}
accuracy = test_model_accuracy(model, directory, truth)

# Save Model if above 90%
if accuracy > 90:
    saveto = 'models/testing/detection_model_test_0.h5'
    num = 1
    while Path(saveto).exists():
        saveto = f'models/testing/detection_model_test_{num}.h5'
        num += 1

    # Save the model
    model.save(saveto)



