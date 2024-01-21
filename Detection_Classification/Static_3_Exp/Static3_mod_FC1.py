
from Investigations.DL_CNN.build_model import build_model
import Static3_mod_config as model_config

if __name__ == '__main__':

    feature_type = 'feature_combo_1'
    feature_params = 'None'

    for len_val in model_config.length:
        build_model(model_config.filepath,
                    len_val,
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
