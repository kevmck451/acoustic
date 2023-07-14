

from Detection.models.template_test_model import test_model_accuracy
from Detection.models.template_load_data import load_audio_data
from Detection.models.template_save_model import save_model
from Detection.models.dataset_info import *

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from pathlib import Path
import numpy as np


# Train Spectral_Model
def Train_Spectral_Detect_Model(dataset, sample_length, name, load_data=False, **kwargs):

    # -------- Load and preprocess data
    if load_data:
        X = np.load('features.npy')
        y = np.load('labels.npy')
    else:
        X, y = load_audio_data(dataset, length=sample_length)
        np.save('features.npy', X)
        np.save('labels.npy', y)

    # print(X.shape)
    # print(y.shape)

    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)
    l2_value = kwargs.get('l2_value', 0.01)
    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'binary_crossentropy')
    metric = kwargs.get('metric', 'accuracy')
    patience = kwargs.get('patience', 6)
    epochs = kwargs.get('epochs', 50)
    batch_size = kwargs.get('batch_size', 32)


    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Test accuracy of Model
    accuracy = test_model_accuracy(model, directory_test_1, sample_length)

    # Save Model if above 90%
    # if accuracy[0] >= 90:
    #     save_model(model, 'detect', 'spec', sample_length, accuracy[0])

    save_model(model, 'detect', name, sample_length, accuracy[0])

if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')
    sample_length = 5
    name = 'spec'
    test_size = 0.2
    random_state = 42
    l2_value = 0.01
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metric = 'accuracy'
    patience = 4
    epochs = 50
    batch_size = 32

    # Train first time to save feature values
    Train_Spectral_Detect_Model(dataset, sample_length, load_data=False,
                                test_size=test_size,
                                random_state=random_state,
                                l2_value=l2_value,
                                optimizer=optimizer,
                                loss=loss,
                                metric=metric,
                                patience=patience,
                                epochs=epochs,
                                batch_size=batch_size
                                )
    while True:
        # Train using load save data to save time
        Train_Spectral_Detect_Model(dataset, sample_length, load_data=True,
                                    test_size=test_size,
                                    random_state=random_state,
                                    l2_value=l2_value,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metric=metric,
                                    patience=patience,
                                    epochs=epochs,
                                    batch_size=batch_size
                                    )


