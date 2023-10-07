
from Detection_Classification.CNN_Models.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2')
    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Engine vs Ambience/test'

    sample_lengths = [10, 8, 6, 4, 2]
    feature_types = ['spectral', 'mfcc', 'filter1']
    model_types = ['basic_1', 'basic_2', 'deep_1', 'deep_2', 'deep_3']
    # feature_params = (70, 3000)     # Spectrum
    feature_params = 100           # MFCC

    Train_Detect_Model(dataset,
                       sample_lengths[2],
                       feature_types[1],
                       model_types[4],
                       testing_path,
                       load_data=True,
                       feature_params=feature_params)

