from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path


if __name__ == '__main__':
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')
    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Static Detection/dataset'

    sample_length = 5
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
                       sample_length,
                       'spectral',
                       model_types[1],
                       specs,
                       testing_path,
                       load_data=True,
                       display=True)