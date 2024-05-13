
from Investigations.DL_CNN.build_model import build_model
import model_config

def train_spectrogram_model():
    feature_type = 'spectral'
    feature_params = {'bandwidth': (70, 3000), 'nperseg': 8192}

    build_model(model_config.filepath,
                model_config.length,
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

if __name__ == '__main__':
    train_spectrogram_model()
