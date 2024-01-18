
from Investigations.DL_CNN.build_model import build_model


if __name__ == '__main__':

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/' \
               '1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 6'

    # Loading Features
    length = [2, 4, 6, 8, 10]
    sample_rate = [12_000, 18_000, 24_000, 36_000, 48_000]
    multi_channel = ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
    chunk_type = ['regular', 'window']
    process_list = ['normalize']  # add labels to list in order to create new processing chain
    feature_type = 'mfcc'

    # Create Model
    conv_layers = [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))]
    dense_layers = [512, 256, 128]
    l2_value = 0.01
    dropout_rate = 0.5
    activation = 'relu' # 'elu' or 'relu'

    # Train Model
    test_size = 0.2
    random_state = 42
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metric = 'accuracy'
    patience = 8
    epochs = 50
    batch_size = 48

    # feature_params = {'n_coeffs': 40}  # MFCC
    # build_model(filepath, length[2], sample_rate[2], multi_channel[0], chunk_type,
    #              process_list, feature_type, feature_params, conv_layers, dense_layers, l2_value,
    #              dropout_rate, activation,test_size, random_state, optimizer, loss, metric, patience, epochs, batch_size)

    feature_params_list = [{'n_coeffs': 50}, {'n_coeffs': 70}, {'n_coeffs': 90}]
    for feature_params in feature_params_list:
        build_model(filepath, length[2], sample_rate[2], multi_channel[0], chunk_type, process_list, feature_type, feature_params,
                    conv_layers, dense_layers, l2_value, dropout_rate, activation,
                    test_size, random_state, optimizer, loss, metric, patience, epochs, batch_size)

