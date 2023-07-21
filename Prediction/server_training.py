
from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    # create a thread for function with arguments
    Train_Detect_Model(dataset, 10, 'spectral', 'basic', False)
    Train_Detect_Model(dataset, 5, 'spectral', 'basic', False)
    Train_Detect_Model(dataset, 2, 'spectral', 'basic', False)
    Train_Detect_Model(dataset, 10, 'filter1', 'basic', False)
    Train_Detect_Model(dataset, 5, 'filter1', 'basic', False)
    Train_Detect_Model(dataset, 2, 'filter1', 'basic', False)

    while True:
        Train_Detect_Model(dataset, 10, 'spectral', 'basic', True)
        Train_Detect_Model(dataset, 5, 'spectral', 'basic', True)
        Train_Detect_Model(dataset, 2, 'spectral', 'basic', True)
        Train_Detect_Model(dataset, 10, 'filter1', 'basic', True)
        Train_Detect_Model(dataset, 5, 'filter1', 'basic', True)
        Train_Detect_Model(dataset, 2, 'filter1', 'basic', True)
        Train_Detect_Model(dataset, 10, 'spectral', 'deep', True)
        Train_Detect_Model(dataset, 5, 'spectral', 'deep', True)
        Train_Detect_Model(dataset, 2, 'spectral', 'deep', True)
        Train_Detect_Model(dataset, 10, 'filter1', 'deep', True)
        Train_Detect_Model(dataset, 5, 'filter1', 'deep', True)
        Train_Detect_Model(dataset, 2, 'filter1', 'deep', True)


