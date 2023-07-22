

from Prediction.base_files.Spectral_Model_10s.accuracy.test_model_accuracy import test_model_accuracy
from Prediction.base_files.Spectral_Model_10s.Spectral_feature_extraction import extract_features
from Prediction.base_files.Spectral_Model_10s.accuracy.generate_truth import generate_truth
from Prediction.dataset_info import *

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from pathlib import Path
import numpy as np


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

# Train Spectral_Model_10s
def spectral_detection_model(load_data=False):
    # Path to audio samples
    static_dataset = Path(
        '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/dataset')

    # -------- Load and preprocess data
    if load_data:
        X = np.load('features.npy')
        y = np.load('labels.npy')
    else:
        X, y = load_audio_data(static_dataset)
        np.save('features.npy', X)
        np.save('labels.npy', y)

    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    model.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Test accuracy of Model
    truth = generate_truth(directory_test_1)
    accuracy = test_model_accuracy(model, directory_test_1, truth)

    # Save Model if above 90%
    if accuracy > 90:
        saveto = f'testing/spec_detect_model_{accuracy}.h5'
        num = 1
        while Path(saveto).exists():
            saveto = f'testing/spec_detect_model_{accuracy}_{num}.h5'
            num += 1

        # Save the model
        model.save(saveto)


if __name__ == '__main__':

    while True:
        spectral_detection_model(load_data=True)


