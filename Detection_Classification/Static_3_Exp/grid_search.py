
from Investigations.DL_CNN.build_model import build_model
from Static3_mod_detector import custom_detector_t1_every_sample
from Static3_mod_detector import custom_detector_t2_every_sample
import Static3_mod_config as model_config

from pathlib import Path

if __name__ == '__main__':

    feature_type = 'spectral'
    feature_params = {'bandwidth': (350, 3000), 'nperseg': 16384}
    length = 6

    conv_layers = [(32, (3, 3)), (64, (3, 3))]
    dense_layers = [256, 128]
    l2_value = 0.01
    dropout_rate = 0.5
    activation = 'relu'

    conv_list = [
        [(16, (2, 2)), (32, (2, 2))],
        [(16, (3, 3)), (32, (3, 3))],
        [(16, (4, 4)), (32, (4, 4))],
        [(16, (5, 5)), (32, (5, 5))],
        [(32, (2, 2)), (64, (2, 2))],
        [(32, (3, 3)), (64, (3, 3))],
        [(32, (4, 4)), (64, (4, 4))],
        [(32, (5, 5)), (64, (5, 5))],
        [(32, (3, 3)), (64, (3, 3))],
        [(64, (2, 2)), (128, (2, 2))],
        [(64, (3, 3)), (128, (3, 3))],
        [(64, (4, 4)), (128, (4, 4))],
        [(64, (5, 5)), (128, (5, 5))],
        [(128, (2, 2)), (256, (2, 2))],
        [(128, (3, 3)), (256, (3, 3))],
        [(128, (4, 4)), (256, (4, 4))],
        [(128, (5, 5)), (256, (5, 5))],
        [(16, (2, 2)), (32, (2, 2)), (64, (2, 2))],
        [(16, (3, 3)), (32, (3, 3)), (64, (3, 3))],
        [(16, (4, 4)), (32, (4, 4)), (64, (4, 4))],
        [(16, (5, 5)), (32, (5, 5)), (64, (5, 5))],
        [(32, (2, 2)), (64, (2, 2)), (128, (2, 2))],
        [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))],
        [(32, (4, 4)), (64, (4, 4)), (128, (4, 4))],
        [(32, (5, 5)), (64, (5, 5)), (128, (5, 5))],
    ]

    conv_list_2 = [
        [(16, (2, 2)), (32, (2, 2)), (64, (2, 2))],
        [(16, (3, 3)), (32, (3, 3)), (64, (3, 3))],
        [(16, (4, 4)), (32, (4, 4)), (64, (4, 4))],
        [(16, (5, 5)), (32, (5, 5)), (64, (5, 5))],
        [(32, (2, 2)), (64, (2, 2)), (128, (2, 2))],
        [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))],
        [(32, (4, 4)), (64, (4, 4)), (128, (4, 4))],
        [(32, (5, 5)), (64, (5, 5)), (128, (5, 5))],
    ]

    model_list = []

    for hyperparameter in conv_list:
        build_model(model_config.filepath,
                    length,
                    model_config.sample_rate,
                    model_config.multi_channel,
                    model_config.chunk_type,
                    model_config.process_list,
                    feature_type,
                    feature_params,
                    hyperparameter,
                    model_config.dense_layers,
                    model_config.l2_value,
                    model_config.dropout_rate,
                    model_config.activation,
                    model_config.test_size,
                    model_config.random_state,
                    model_config.optimizer,
                    model_config.loss,
                    model_config.metric,
                    model_config.patience,
                    model_config.epochs,
                    model_config.batch_size)

        base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
        test_base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
        save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

        model_path = Path(f'{base_path}/Detection_Classification/Static_3_Exp/model_library')
        test_directory_path = f'{test_base_path}/ML Model Data/Static Test 3/test 1'
        save_directory = f'{save_base_dir}/Engine Hex Static 3'

        for model in model_path.iterdir():
            if 'h5' in str(model):
                if model not in model_list:
                    print(model)
                    model_list.append(model)
                    custom_detector_t1_every_sample(model.resolve(), test_directory_path, save=True, save_path=save_directory)