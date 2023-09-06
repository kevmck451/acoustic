
from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path




if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Classification/dataset')
    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Classification/tests'

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

    # Train_Detect_Model(dataset, sample_lengths[2], feature_types[2], model_types[0], specs, testing_path, load_data=True)
    Train_Detect_Model(dataset, sample_lengths[2], feature_types[0], model_types[0], specs, testing_path, load_data=True)

