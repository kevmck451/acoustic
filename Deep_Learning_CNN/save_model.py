# File for model naming

import json
from pathlib import Path

def save_model(filepath, length, sample_rate, multi_channel, process_list, feature_type, feature_params,
               input_shape, conv_layers, dense_layers, l2_value, dropout_rate, activation,
               test_size, random_state, model, optimizer, loss, metric, patience, epochs, batch_size, runtime):

    index = 0
    model_extension = '.h5'
    text_extension = '.txt'
    lib_dir = f'{Path.cwd()}/model_library'
    save_label_dir_path = Path(lib_dir)
    Path(save_label_dir_path).mkdir(exist_ok=True)

    if feature_type == 'spectral':
        bandwidth = feature_params.get('bandwidth')
        feature_save_name = f'{bandwidth[0]}-{bandwidth[1]}'
    elif feature_type == 'mfcc': feature_save_name = f"{feature_params.get('n_coeffs')}"
    else: feature_save_name = 'None'

    model_layers = len(conv_layers) + len(dense_layers)
    model_type = f'{model_layers}-layers'

    model_saveto = f'{lib_dir}/{feature_type}_{feature_save_name}_{str(length)}s_{model_type}_{str(index)}{model_extension}'

    while Path(model_saveto).exists():
        index += 1
        model_saveto = f'{lib_dir}/{feature_type}_{feature_save_name}_{str(length)}s_{model_type}_{str(index)}{model_extension}'

    model.save(model_saveto)

    # Save text file with all info about model
    text_saveto = f'{lib_dir}/{feature_type}_{feature_save_name}_{str(length)}s_{model_type}_{str(index)}{text_extension}'

    feat = 'None'
    if feature_type == 'spectral':
        bandwidth = feature_params.get('bandwidth')
        window = feature_params.get('window_size')
        hop_size = feature_params.get('hop_size')
        feat = f'Bandwidth: ({bandwidth[0]}-{bandwidth[1]}) / Window Size: {window} / Hop Size: {hop_size}'
    if feature_type == 'mfcc':
        feat = feature_params.get('n_coeffs')
        feat = f'Num Coeffs: {feat}'

    # model.summary()
    optimizer_config = model.optimizer.get_config()

    feat_type = f'Feature Type: {feature_type.upper()}'
    # params = f'Feature Parameters: {feat}'
    params = f'Feature Parameters: {feature_params}'
    sr = f'Sample Rate: {sample_rate} Hz'
    leng = f'Sample Length: {length} sec'
    shape = f'Shape: {input_shape}'
    multch = f'Multi Channel: {multi_channel.title()}'
    path = f'Filepath: {filepath}'
    pro_list = f'Process Applied: {process_list}'
    conv_lay = f'Convolutional Layers: {conv_layers}'
    den_lay = f'Dense Layers: {dense_layers}'
    l2_val = f'Kernal Reg-l2 Value: {l2_value}'
    drop_rate = f'Dropout Rate: {dropout_rate}'
    act_funct = f'Activation Function: {activation}'
    test_siz = f'Test Data Size: {test_size}'
    rand_st = f'Random State: {random_state}'
    opt = f'Optimizer: {optimizer}'
    los = f'Loss: {loss}'
    met = f'Metric: {metric}'
    pat = f'Patience: {patience}'
    ep = f'Epochs: {epochs}'
    bs = f'Batch Size: {batch_size}'
    bt = f'Build Time: {runtime}'
    opt_con = f'Model Config File: {optimizer_config}'


    filenames = [path, multch, sr, leng, pro_list, feat_type, params, shape, conv_lay, den_lay, l2_val, drop_rate, act_funct, test_siz,
                 rand_st, opt, los, met, pat, ep, bs, bt, opt_con]

    with open(text_saveto, 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')

    if Path(model_saveto).exists(): print('Model Save Successful')
    else: print('Model Save Not Successful')

    if Path(text_saveto).exists(): print('Text Save Successful')
    else: print('Text Save Not Successful')



