# file to train model on a dataset


from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
import numpy as np



def train_model(features, labels, test_size, random_state, model, optimizer, loss, metric, patience, epochs,
                batch_size):
    print('Training Model')

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    train_gen = data_generator(X_train, y_train, batch_size)
    val_gen = data_generator(X_test, y_test, batch_size)

    ''' Custom Metrics
    https://keras.io/api/metrics/

    '''

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_test) // batch_size

    model.fit(train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=[early_stopping])

    ''' Visualizing Results:
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    '''

    return model


# This is the data generator function
def data_generator(X, y, batch_size=32):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield np.array(X[start:end]), np.array(y[start:end])