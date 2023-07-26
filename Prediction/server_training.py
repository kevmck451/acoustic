
from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':

    dataset = Path('../../ML Model Data/dataset')

    testing_path = '../../ML Model Data/accuracy/dataset'

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


    for length in sample_lengths:
        for feature in feature_types:
            for model in model_types:
                Train_Detect_Model(dataset, length, feature, model, specs, testing_path, load_data=False)







