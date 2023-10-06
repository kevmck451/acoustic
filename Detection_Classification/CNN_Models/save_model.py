# File for model naming

import json
from pathlib import Path

def save_model(model, model_type, feature, input_time, accuracy, specs, runtime, **kwargs):
    index = 0
    model_extension = '.h5'
    text_extension = '.txt'
    lib_dir = f'{Path.cwd()}/Prediction/model_library'

    feature_params = kwargs.get('feature_params', 'None')
    if feature == 'spectral' and feature_params != 'None': feature_save_name = f'{feature_params[0]}-{feature_params[1]}'
    elif feature == 'mfcc' and feature_params != 'None': feature_save_name = f'{feature_params}'
    else: feature_save_name = 'None'

    model_saveto = f'{lib_dir}/{feature}_{feature_save_name}_{str(input_time)}_{model_type}_{accuracy}_{str(index)}{model_extension}'

    while Path(model_saveto).exists():
        index += 1
        model_saveto = f'{lib_dir}/{feature}_{feature_save_name}_{str(input_time)}_{model_type}_{accuracy}_{str(index)}{model_extension}'

    model.save(model_saveto)

    # Save text file with all info about model
    text_saveto = f'{lib_dir}/{feature}_{feature_save_name}_{str(input_time)}_{model_type}_{accuracy}_{str(index)}{text_extension}'

    feature_params = kwargs.get('feature_params', 'None')
    with open(text_saveto, 'w') as f:
        f.write('Model Type: ' + model_type + '\n')
        f.write('Feature: ' + feature + '\n')
        f.write('Feature Parameters: ' + feature_params + '\n')
        f.write('Sample Length: ' + str(input_time) + 's\n')
        f.write('Accuracy: ' + str(accuracy) + '%\n')
        f.write('Total Runtime: ' + str(runtime) + 's\n')
        f.write('Specs: ' + json.dumps(specs, indent=4) + '\n')

    if Path(model_saveto).exists(): print('Model Save Successful')
    else: print('Model Save Not Successful')

    if Path(text_saveto).exists(): print('Text Save Successful')
    else: print('Text Save Not Successful')



