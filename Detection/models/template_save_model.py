# File for model naming

from pathlib import Path


def save_model(model, model_type, feature, input_time, accuracy):
    index = 0
    extension = '.h5'
    lib_dir = 'Spectral_Model_10s/model_library'

    saveto = f'{lib_dir}/{model_type}_{feature}_{str(input_time)}_{accuracy}_{str(index)}{extension}'

    while Path(saveto).exists():
        saveto = f'{lib_dir}/{model_type}_{feature}_{str(input_time)}_{accuracy}_{str(index)}{extension}'
        index += 1

    model.save(saveto)

    if Path(saveto).exists(): print('Save Successful')
    else: print('Save Not Successful')

