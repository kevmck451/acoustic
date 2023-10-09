# file to train model on a dataset


from Deep_Learning_CNN.load_features import load_features
from Deep_Learning_CNN.create_models import create_model
from Acoustic.utils import time_class


from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from pathlib import Path
import numpy as np





def train_model(features, labels, test_size, random_state, model, optimizer, loss, metric, patience, epochs, batch_size):
    print('Training Model')

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)


    # gc.collect()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stopping])


    return model

