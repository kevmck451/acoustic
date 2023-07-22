
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from pathlib import Path
import numpy as np


# Basic CNN: 3 layers
def basic_model_1(features, labels, **kwargs):
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
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
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    return model

# Basic CNN: 3 layers
def basic_model_2(features, labels, **kwargs):
    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)
    l2_value = kwargs.get('l2_value', 0.01)
    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'binary_crossentropy')
    metric = kwargs.get('metric', 'accuracy')
    patience = kwargs.get('patience', 3)
    epochs = kwargs.get('epochs', 20)
    batch_size = kwargs.get('batch_size', 12)

    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='sigmoid', kernel_regularizer=l2(l2_value)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    return model

# Deep CNN: 4 layers
def deep_model_1(features, labels, **kwargs):

    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)
    l2_value = kwargs.get('l2_value', 0.01)
    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'binary_crossentropy')
    metric = kwargs.get('metric', 'accuracy')
    patience = kwargs.get('patience', 3)
    epochs = kwargs.get('epochs', 20)
    batch_size = kwargs.get('batch_size', 12)

    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    return model

# Deep CNN: 4 layers
def deep_model_2(features, labels, **kwargs):

    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)
    l2_value = kwargs.get('l2_value', 0.01)
    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'binary_crossentropy')
    metric = kwargs.get('metric', 'accuracy')
    patience = kwargs.get('patience', 3)
    epochs = kwargs.get('epochs', 20)
    batch_size = kwargs.get('batch_size', 12)

    # Create a Model for Training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    return model