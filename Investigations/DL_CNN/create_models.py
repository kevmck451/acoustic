

import keras
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.regularizers import l2
import os
import random




# base method for creating models
def create_model(input_shape, conv_layers, dense_layers, l2_value=0.01, dropout_rate=0.5, activation='relu'):
    """
    Parameters:
    - input_shape: tuple, shape of the input data (height, width, channels)
    - conv_layers: list of tuples, each tuple has (number of filters, kernel size)
    - dense_layers: list of integers, number of neurons in each dense layer
    - dropout_rate: dropout rate for regularization
    - activation: activation function to use

    Returns: model: Keras Sequential model
    """
    print('Creating Model')

    # Code to make models more deterministic
    keras.utils.set_random_seed(1337)
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(1337)

    model = Sequential()

    i = 0

    # First convolutional layer with input shape
    filters, kernel_size = conv_layers[0]
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    # model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(AveragePooling2D(pool_size=(1, 1)))
    model.add(Dropout(dropout_rate))
    i += 1

    # Adding subsequent convolutional layers
    for filters, kernel_size in conv_layers[1:]:
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation))

        if i%2 == 0:
            # model.add(MaxPooling2D(pool_size=(1, 1)))
            model.add(AveragePooling2D(pool_size=(1, 1)))
            i += 1
        else:
            # model.add(MaxPooling2D(pool_size=(2, 1)))
            model.add(AveragePooling2D(pool_size=(2, 1)))
            i += 1

        model.add(Dropout(dropout_rate))

    # Flatten the features for dense layers
    model.add(Flatten())

    # Adding dense layers with dropout for regularization
    for units in dense_layers:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))

    # Final layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model


