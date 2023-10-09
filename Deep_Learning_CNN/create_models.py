


from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2


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

    model = Sequential()

    # First convolutional layer with input shape
    filters, kernel_size = conv_layers[0]
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding subsequent convolutional layers
    for filters, kernel_size in conv_layers[1:]:
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the features for dense layers
    model.add(Flatten())

    # Adding dense layers with dropout for regularization
    for units in dense_layers:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))

    # Final layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    return model


