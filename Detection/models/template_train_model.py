

from Detection.models.test_model import test_model_accuracy
from Detection.models.create_model import basic_model_1
from Detection.models.create_model import deep_model_1
from Detection.models.load_data import load_audio_data
from Detection.models.save_model import save_model
from Detection.models.dataset_info import *
from Acoustic.utils import time_class

from pathlib import Path
import numpy as np



# Train Spectral_Model
def Train_Detect_Model(dataset, sample_length, feature_type, model_type, load_data=False):

    timing_stats = time_class(name='Model Training')

    # -------- Load and preprocess data
    if load_data:
        try:
            features = np.load(f'features_{feature_type}_{sample_length}s.npy')
            labels = np.load(f'labels_{feature_type}_{sample_length}s.npy')
        except:
            features, labels = load_audio_data(dataset, sample_length, feature_type)
            np.save(f'features_{feature_type}_{sample_length}s.npy', features)
            np.save(f'labels_{feature_type}_{sample_length}s.npy', labels)
    else:
        features, labels = load_audio_data(dataset, sample_length, feature_type)
        np.save(f'features_{feature_type}_{sample_length}s.npy', features)
        np.save(f'labels_{feature_type}_{sample_length}s.npy', labels)

    # print(X.shape)
    # print(y.shape)

    # Create a Model for Training
    test_size = 0.2
    random_state = 42
    l2_value = 0.01
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metric = 'accuracy'
    patience = 4
    epochs = 50
    batch_size = 32

    if model_type == 'basic':
        model = basic_model_1(features, labels, test_size=test_size,
                             random_state=random_state, l2_value=l2_value,
                             optimizer=optimizer, loss=loss,
                             metric=metric, patience=patience,
                             epochs=epochs, batch_size=batch_size)

    elif model_type == 'deep':
        model = deep_model_1(features, labels, test_size=test_size,
                              random_state=random_state, l2_value=l2_value,
                              optimizer=optimizer, loss=loss,
                              metric=metric, patience=patience,
                              epochs=epochs, batch_size=batch_size)

    else:
        model = basic_model_1(features, labels, test_size=test_size,
                              random_state=random_state, l2_value=l2_value,
                              optimizer=optimizer, loss=loss,
                              metric=metric, patience=patience,
                              epochs=epochs, batch_size=batch_size)

    # Test accuracy of Model
    accuracy = test_model_accuracy(model, directory_test_1, sample_length, feature_type)

    # Save Model
    save_model(model, model_type, feature_type, sample_length, accuracy[0])
    # if accuracy[0] >= 90:
    #     save_model(model, 'detect', 'spec', sample_length, accuracy[0])

    timing_stats.stats()

if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')
    sample_length = 5
    feature_type = 'filter1' #'spectral'
    model_type = 'deep'

    while True:
        Train_Detect_Model(dataset, sample_length, feature_type, model_type, load_data=True)

