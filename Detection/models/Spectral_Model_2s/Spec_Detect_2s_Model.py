

from Detection.models.Spectral_Model_2s.accuracy.test_mod_acc_2s import test_model_accuracy
from Detection.models.Spectral_Model_2s.Spec_Detect_FE_2s import load_audio_data
from Detection.models.model_saving import save_model
from Detection.models.dataset_info import *

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from pathlib import Path
import numpy as np



# Train Spectral_Model_10s
def spectral_detection_model(load_data=False):
    # Path to audio samples
    # dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/dataset')
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    # -------- Load and preprocess data
    if load_data:
        X = np.load('features.npy')
        y = np.load('labels.npy')
    else:
        X, y = load_audio_data(dataset)
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
    accuracy = test_model_accuracy(model, directory_test_1)

    # Save Model if above 90%
    if accuracy[0] >= 96:
        save_model(model, 'detect', 'spec', 2, accuracy[0])



if __name__ == '__main__':

    while True:
        spectral_detection_model(load_data=True)


