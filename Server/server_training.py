
from Detection_Classification.CNN_Models.template_train_model import Train_Detect_Model

from pathlib import Path

if __name__ == '__main__':

    base_dir = '/home/kmcknze1'
    dataset = Path(f'{base_dir}/Data/Engine vs Ambience/dataset 2')
    testing_path = Path(f'{base_dir}/Data/Engine vs Ambience/test')

    sample_lengths = [10, 8, 6, 4, 2]
    feature_types = ['spectral', 'mfcc', 'filter1']
    model_types = ['basic_1', 'basic_2', 'deep_1', 'deep_2', 'deep_3']
    # feature_params = (70, 3000)     # Spectrum
    feature_params = 120  # MFCC

    Train_Detect_Model(dataset,
                       sample_lengths[2],
                       feature_types[1],
                       model_types[0],
                       testing_path,
                       load_data=False,
                       feature_params=feature_params)


'''
ssh kmcknze1@c2-kevin.uom.memphis.edu

password: m2d2jkl9123

tmux

conda activate acoustic1

cd acoustic

git pull

python3 -m CNN_Models.server_training
'''
