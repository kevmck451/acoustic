# File for model naming

import json
from pathlib import Path

def save_model(model, model_type, feature, input_time, accuracy, specs, runtime):
    index = 0
    model_extension = '.h5'
    text_extension = '.txt'
    lib_dir = '/model_library'

    model_saveto = f'{lib_dir}/{model_type}_{feature}_{str(input_time)}_{accuracy}_{str(index)}{model_extension}'

    while Path(model_saveto).exists():
        index += 1
        model_saveto = f'{lib_dir}/{model_type}_{feature}_{str(input_time)}_{accuracy}_{str(index)}{model_extension}'

    model.save(model_saveto)

    # Save text file with all info about model
    text_saveto = f'{lib_dir}/{model_type}_{feature}_{str(input_time)}_{accuracy}_{str(index)}{text_extension}'

    with open(text_saveto, 'w') as f:
        f.write('Model Type: ' + model_type + '\n')
        f.write('Feature: ' + feature + '\n')
        f.write('Input Time: ' + str(input_time) + '\n')
        f.write('Accuracy: ' + str(accuracy) + '\n')
        f.write('Total Runtime: ' + str(runtime) + '\n')
        f.write('Specs: ' + json.dumps(specs, indent=4) + '\n')

    if Path(model_saveto).exists(): print('Model Save Successful')
    else: print('Model Save Not Successful')

    if Path(text_saveto).exists(): print('Text Save Successful')
    else: print('Text Save Not Successful')



