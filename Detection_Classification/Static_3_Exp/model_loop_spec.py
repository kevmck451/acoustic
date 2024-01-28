
from Investigations.DL_CNN.build_model import build_model
from Static3_mod_detector import custom_detector_t1_every_sample
from Static3_mod_detector import custom_detector_t2_every_sample
import Static3_mod_config as model_config

from pathlib import Path

if __name__ == '__main__':

    feature_type = 'spectral'
    feature_params = {'bandwidth': (70, 9000), 'nperseg': 16384}
    length = 6
    model_list = []

    for i in range(2):
        build_model(model_config.filepath,
                    length,
                    model_config.sample_rate,
                    model_config.multi_channel,
                    model_config.chunk_type,
                    model_config.process_list,
                    feature_type,
                    feature_params,
                    model_config.conv_layers,
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
        test_directory_path = f'{test_base_path}/ML Model Data/Static Test 3/test 2'
        save_directory = f'{save_base_dir}/Engine Hex Static 3'

        for model in model_path.iterdir():
            if 'h5' in str(model):
                if model not in model_list:
                    print(model)
                    model_list.append(model)
                    custom_detector_t2_every_sample(model.resolve(), test_directory_path, save=True, save_path=save_directory)