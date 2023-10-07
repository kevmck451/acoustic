

from Detection_Classification.CNN_Models.test_model import test_model_accuracy
import Detection_Classification.CNN_Models.create_model as create_model
from Detection_Classification.CNN_Models.load_data import load_audio_data
from Detection_Classification.CNN_Models.save_model import save_model
from Acoustic.utils import time_class

from pathlib import Path
import numpy as np

# Train Spectral_Model
def Train_Detect_Model(dataset, sample_length, feature_type, model_type, test_path, load_data=False, display=False, **kwargs):
    timing_stats = time_class(name='Model Training')
    feature_params = kwargs.get('feature_params', 'None')

    if feature_type == 'spectral' and feature_params != 'None': feature_save_name = f'{feature_params[0]}-{feature_params[1]}'
    elif feature_type == 'mfcc' and feature_params != 'None': feature_save_name = f'{feature_params}'
    else: feature_save_name = 'None'

    # -------- Load and preprocess data
    save_path = f'{Path.cwd()}/Prediction/features_labels'
    if load_data:
        try:
            features = np.load(f'{save_path}/features_{feature_type}_{feature_save_name}_{sample_length}s.npy')
            labels = np.load(f'{save_path}/labels_{feature_type}_{feature_save_name}_{sample_length}s.npy')
        except:
            features, labels = load_audio_data(dataset, sample_length, feature_type, feature_params=feature_params)
            np.save(f'{save_path}/features_{feature_type}_{feature_save_name}_{sample_length}s.npy', features)
            np.save(f'{save_path}/labels_{feature_type}_{feature_save_name}_{sample_length}s.npy', labels)
    else:
        features, labels = load_audio_data(dataset, sample_length, feature_type, feature_params=feature_params)
        np.save(f'{save_path}/features_{feature_type}_{feature_save_name}_{sample_length}s.npy', features)
        np.save(f'{save_path}/labels_{feature_type}_{feature_save_name}_{sample_length}s.npy', labels)

    # print(X.shape)
    # print(y.shape)

    specs_default = {
        'test_size': 0.2,
        'random_state': 42,
        'l2_value': 0.01,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metric': 'accuracy',
        'patience': 5,
        'epochs': 50,
        'batch_size': 24}

    specs = kwargs.get('specs', specs_default)

    if model_type == 'basic_1':
        model = create_model.basic_model_1(features, labels, test_size=specs['test_size'],
                                           random_state=specs['random_state'], l2_value=specs['l2_value'],
                                           optimizer=specs['optimizer'], loss=specs['loss'],
                                           metric=specs['metric'], patience=specs['patience'],
                                           epochs=specs['epochs'], batch_size=specs['batch_size'])
    elif model_type == 'basic_2':
        model = create_model.basic_model_2(features, labels, test_size=specs['test_size'],
                                          random_state=specs['random_state'], l2_value=specs['l2_value'],
                                          optimizer=specs['optimizer'], loss=specs['loss'],
                                          metric=specs['metric'], patience=specs['patience'],
                                          epochs=specs['epochs'], batch_size=specs['batch_size'])
    elif model_type == 'deep_1':
        model = create_model.deep_model_1(features, labels, test_size=specs['test_size'],
                                          random_state=specs['random_state'], l2_value=specs['l2_value'],
                                          optimizer=specs['optimizer'], loss=specs['loss'],
                                          metric=specs['metric'], patience=specs['patience'],
                                          epochs=specs['epochs'], batch_size=specs['batch_size'])
    elif model_type == 'deep_2':
        model = create_model.deep_model_2(features, labels, test_size=specs['test_size'],
                                          random_state=specs['random_state'], l2_value=specs['l2_value'],
                                          optimizer=specs['optimizer'], loss=specs['loss'],
                                          metric=specs['metric'], patience=specs['patience'],
                                          epochs=specs['epochs'], batch_size=specs['batch_size'])
    elif model_type == 'deep_3':
        model = create_model.deep_model_3(features, labels, test_size=specs['test_size'],
                                          random_state=specs['random_state'], l2_value=specs['l2_value'],
                                          optimizer=specs['optimizer'], loss=specs['loss'],
                                          metric=specs['metric'], patience=specs['patience'],
                                          epochs=specs['epochs'], batch_size=specs['batch_size'])
    else:
        model = create_model.basic_model_1(features, labels, test_size=specs['test_size'],
                                           random_state=specs['random_state'], l2_value=specs['l2_value'],
                                           optimizer=specs['optimizer'], loss=specs['loss'],
                                           metric=specs['metric'], patience=specs['patience'],
                                           epochs=specs['epochs'], batch_size=specs['batch_size'])

    # Test accuracy of Model
    if display:
        accuracy = test_model_accuracy(model, test_path, sample_length, feature_type, display=True, feature_params=feature_params)
    else:
        accuracy = test_model_accuracy(model, test_path, sample_length, feature_type, feature_params=feature_params)

    total_runtime = timing_stats.stats()

    # Save Model
    save_model(model, model_type, feature_type, sample_length, accuracy[0], specs, total_runtime, feature_params=feature_params)
    # if accuracy[0] >= 90:
    #     save_model(model, 'detect', 'spec', sample_length, accuracy[0])


if __name__ == '__main__':
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Engine vs Nothing/dataset')
    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Engine vs Nothing/test'

    sample_lengths = [10, 8, 6, 4, 2]
    feature_types = ['spectral', 'filter1', 'mfcc']
    model_types = ['basic_1', 'basic_2', 'deep_1', 'deep_2']

    specs = {
        'test_size': 0.2,
        'random_state': 42,
        'l2_value': 0.01,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metric': 'accuracy',
        'patience': 5,
        'epochs': 50,
        'batch_size': 24}

    Train_Detect_Model(dataset,
                       sample_lengths[2],
                       feature_types[2],
                       model_types[0],
                       testing_path,
                       load_data=True,
                       specs=specs)

