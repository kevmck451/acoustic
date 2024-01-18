
from Investigations.Convo_Neural_Network.load_features import load_features
from Investigations.Convo_Neural_Network.create_models import create_model
from Investigations.Convo_Neural_Network.train_model import train_model
from Investigations.Convo_Neural_Network.save_model import save_model
from Acoustic.utils import time_class


from pathlib import Path


def build_model(filepath, length, sample_rate, multi_channel, chunk_type, process_list, feature_type, feature_params,
                conv_layers, dense_layers, l2_value, dropout_rate, activation,
                test_size, random_state, optimizer, loss, metric, patience, epochs, batch_size):

    timing_stats = time_class(name='Build Model')

    filepath = Path(filepath)

    features, labels = load_features(filepath, length, sample_rate, multi_channel, chunk_type, process_list, feature_type, feature_params)

    # Create a flexible model
    input_shape = features.shape[1:]
    model = create_model(input_shape, conv_layers, dense_layers, l2_value, dropout_rate, activation)

    # Train Model
    model = train_model(features, labels, test_size, random_state, model, optimizer, loss, metric, patience, epochs, batch_size)

    # accuracy = test_model()

    total_runtime = timing_stats.stats()

    # Save Model
    save_model(filepath, length, sample_rate, multi_channel, chunk_type, process_list, feature_type, feature_params, input_shape, conv_layers, dense_layers,
               l2_value, dropout_rate, activation, test_size, random_state, model, optimizer, loss, metric, patience, epochs, batch_size, total_runtime)
    # if accuracy[0] >= 90:
    #     save_model(model, 'detect', 'spec', sample_length, accuracy[0])



if __name__ == '__main__':
    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/' \
                    '1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2'

    # Loading Features
    length = [2, 4, 6, 8, 10]
    sample_rate = [12_000, 18_000, 24_000, 36_000, 48_000]
    multi_channel = ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
    chunk_type = ['regular', 'window']
    process_list = ['normalize']  # add labels to list in order to create new processing chain
    feature_type = ['spectral', 'mfcc', 'feature_combo_1']
    window_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    hop_sizes = [128, 256, 512, 1024]
    feature_params = {'bandwidth':(70, 6000), 'window_size':window_sizes[4], 'hop_size':hop_sizes[2]}  # Spectrum
    # feature_params = {'n_coeffs': 13}  # MFCC

    # Create Model
    conv_layers = [(32, (3, 3)), (64, (3, 3))]
    dense_layers = [128, 64]
    l2_value = 0.01
    dropout_rate = 0.5
    activation = 'relu'

    # Train Model
    test_size = 0.2
    random_state = 42
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metric = 'accuracy'
    patience = 10
    epochs = 50
    batch_size = 24

    build_model(filepath, length[4], sample_rate[2], multi_channel[0], chunk_type, process_list, feature_type[0],
                feature_params, conv_layers, dense_layers, l2_value, dropout_rate, activation,
                test_size, random_state, optimizer, loss, metric, patience, epochs, batch_size)

