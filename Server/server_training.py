
from Prediction.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':

    base_dir = '/home/kmcknze1'
    dataset = Path(f'{base_dir}/ML Model Data/dataset')
    testing_path = Path(f'{base_dir}/ML Model Data/accuracy/dataset')

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
        'patience': 10,
        'epochs': 50,
        'batch_size': 12}

    # Train Model using all parameters
    Train_Detect_Model(dataset,
                       sample_lengths[0],
                       feature_types[2],
                       model_types[2],
                       specs,
                       testing_path,
                       load_data=True)






'''
ssh kmcknze1@c2-kevin.uom.memphis.edu

password: m2d2jkl9123

tmux

conda activate acoustic1

cd acoustic

git pull

python3 -m Prediction.server_training
'''
