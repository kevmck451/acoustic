
from Investigations.DL_CNN.build_model import build_model

from pathlib import Path


if __name__ == '__main__':

    filepath = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/'
                    '1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2')

    # Loading Features
    length = [2, 4, 6, 8, 10]
    sample_rate = [12_000, 18_000, 24_000, 36_000, 48_000]
    multi_channel = ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
    chunk_type = ['regular', 'window']
    process_list = ['normalize']  # add labels to list in order to create new processing chain
    feature_type = ['spectral', 'mfcc']
    window_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # Create Model
    conv_layers = [(32, (3, 3)), (64, (3, 3))]
    dense_layers = [128, 64]
    l2_value = 0.01
    dropout_rate = 0.5
    activation = 'relu' # 'relu'

    # Train Model
    test_size = 0.2
    random_state = 42
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metric = 'accuracy'
    patience = 15
    epochs = 50
    batch_size = 24


    # Spectral Model
    feature_params = {'bandwidth' :(70, 20000), 'window_size' :window_sizes[4]}  # Spectrum
    build_model(filepath, length[4], sample_rate[4], multi_channel[0], chunk_type, process_list, feature_type[0], feature_params,
                conv_layers, dense_layers, l2_value, dropout_rate, activation,
                test_size, random_state, optimizer, loss, metric, patience, epochs, batch_size)



