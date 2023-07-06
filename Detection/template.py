

from Acoustic.audio import Audio
from Detection.test_model_accuracy import test_model_accuracy



from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score


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
# Test accuracy of Model
accuracy = test_model_accuracy(model, display=True)

# Save Model if above 90%
if accuracy > 90:
    saveto = 'models/testing/detection_model_test_0.h5'
    num = 1
    while Path(saveto).exists():
        saveto = f'models/testing/detection_model_test_{num}.h5'
        num += 1

    # Save the model
    model.save(saveto)



