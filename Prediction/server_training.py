
from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    sample_length = [10, 8, 6, 4, 2]

    feature_type = ['spectral', 'filter1', 'mfcc']

    model_type = ['basic_1', 'basic_2', 'deep_1', 'deep_2']

    # Create a Model for Training
    specs = {
        'test_size': 0.2,
        'random_state': 42,
        'l2_value': 0.01,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metric': 'accuracy',
        'patience': 4,
        'epochs': 50,
        'batch_size': 24}


    # create a thread for function with arguments

    for length in sample_length:
        for feature in feature_type:
            for model in model_type:
                Train_Detect_Model(dataset, length, feature, model, specs, load_data=False)







