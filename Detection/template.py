

from Acoustic.audio import Audio


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
    mfccs = audio.mfcc()
    # spectro = audio.spectrogram()

    return mfccs
    # return spectro

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

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=20, batch_size=12, validation_data=(X_test, y_test))


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test), callbacks=[early_stopping])



# Test accuracy of Model
truth = {
    '10m-D-DEIdle_b' : 1,
    '10m-D-TIdle_1_c' : 1,
    'Hex_8_Hover_4_a' : 0,
    'Hex_8_Hover_1_a' : 0,
    '10m-D-TIdle_2_c' : 1,
    'Hex_1_Takeoff_a' : 0
}

y_true = []
y_pred = []

for file in Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/Test').iterdir():
    # Load and preprocess the new audio sample
    feature = extract_features(file, 10)
    feature = np.array([feature])
    feature = feature[..., np.newaxis]

    # Predict class
    y_new_pred = model.predict(feature)
    y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction

    # Retrieve true label
    y_true_class = truth.get(file.stem, None)

    # Skip this file if it's not in our truth dictionary
    if y_true_class is None:
        continue

    # Append to our lists
    y_true.append(y_true_class)
    y_pred.append(y_pred_class)

    percent = y_new_pred[0][0]
    print(f'File: {file.stem} / Percent: {np.round((percent * 100), 2)}%')

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {np.round((accuracy * 100), 2)}%')



# Save Model if Desired
answer = input('Do you want to save this Model? y/n: ')
# ------- Save Model
if answer == 'y':
    saveto = 'models/testing/detection_model_test_0.h5'
    num = 1
    while Path(saveto).exists():
        saveto = f'models/testing/detection_model_test_{num}.h5'
        num += 1

    # Save the model
    model.save(saveto)








