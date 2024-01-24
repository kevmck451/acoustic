# File to test the accuracy of a model against some known samples that were excluded from test data

from Investigations.DL_CNN.load_features import load_features

from keras.models import load_model
from pathlib import Path
import numpy as np




# Function to test ML Models's Accuracy
def test_model_accuracy(model_path, audio_path, chunk_type):
    # Load model and data
    model_info = load_model_text_file(model_path)
    features, labels = load_features(audio_path, model_info.get('Sample Length'), model_info.get('Sample Rate'),
                                     model_info.get('Multi Channel'), chunk_type, model_info.get('Process Applied'),
                                     model_info.get('Feature Type').lower(), model_info.get('Feature Parameters'))

    model = load_model(model_path)
    names = load_audio_name_file(model_path, model_info)

    predictions = []

    for feature in features:
        feature = np.expand_dims(feature, axis=0)
        y_new_pred = model.predict(feature)
        predictions.append(y_new_pred[0][0])

    return names, predictions, labels




# Function to use model text file to get parameter info
def load_model_text_file(model_path):
    model = str(model_path)
    text_path_base = model.split('.')[0]
    text_path = f'{text_path_base}.txt'

    model_info = {}

    with open(text_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)

            # Remove any additional spaces
            key = key.strip()
            value = value.strip()

            # Convert the value into suitable format
            if key in ['Convolutional Layers', 'Dense Layers']:
                model_info[key] = eval(value)
            elif key in ['Feature Parameters', 'Model Config File']:
                model_info[key] = eval(value.replace(' /', ','))
            elif key == 'Shape':
                model_info[key] = tuple(map(int, value.strip('()').split(', ')))
            elif key in ['Sample Rate', 'Sample Length', 'Shape', 'Kernal Reg-l2 Value',
                         'Dropout Rate', 'Test Data Size', 'Random State', 'Epochs', 'Batch Size', 'Build Time']:
                try:
                    model_info[key] = int(value.split()[0])
                except ValueError:
                    model_info[key] = float(value.split()[0])
            else:
                model_info[key] = value

    return model_info

# Function to get audio names from file
def load_audio_name_file(model_path, model_info):
    model_name = Path(model_path).stem
    text_file_name = model_name.split('_')[:-2]
    if model_info.get('Feature Type').lower() == 'mfcc':
        text_file_name = '_'.join(text_file_name)
    elif model_info.get('Feature Type').lower() == 'spectral':
        text_file_name[1] = text_file_name[1] + '-None'
        text_file_name = '_'.join(text_file_name)
    else: text_file_name = '_'.join(text_file_name)

    text_file_name = f'features_labels/{text_file_name}_testing_features_files.txt'
    names_path = f'{Path(model_path).parent.parent}/{text_file_name}'

    with open(names_path, 'r') as file:
        names = file.readlines()

    return names



if __name__ == '__main__':

    pass





