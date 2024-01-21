# File to test the accuracy of a model against some known samples that were excluded from test data

from Investigations.DL_CNN.load_features import load_features

from sklearn.metrics import accuracy_score
from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches



# Function to test ML Models's Accuracy
def test_model_accuracy(model_path, audio_path, chunk_type, **kwargs):
    # Initialize lists for storing data
    predictions_sum = {}
    predictions_count = {}
    y_true = []
    y_names = []

    # Load model and data
    model_path = str(model_path)
    model_info = load_model_text_file(model_path)
    path_model = Path(model_path)
    model_name = path_model.stem

    features, labels = load_features(audio_path, model_info.get('Sample Length'), model_info.get('Sample Rate'),
                                     model_info.get('Multi Channel'), chunk_type, model_info.get('Process Applied'),
                                     model_info.get('Feature Type').lower(), model_info.get('Feature Parameters'))

    print('Testing Model')
    model = load_model(model_path)
    names = load_audio_name_file(model_path, model_info)

    # Iterate over each feature, label, and name
    for feature, label, name in zip(features, labels, names):
        feature = np.expand_dims(feature, axis=0)
        y_new_pred = model.predict(feature)

        # Aggregate predictions
        if name not in predictions_sum:
            predictions_sum[name] = 0
            predictions_count[name] = 0
        predictions_sum[name] += y_new_pred[0][0]
        predictions_count[name] += 1

        # Store true label (assuming each name has a corresponding label)
        if name not in y_true:
            y_true.append(label)
            y_names.append(name)

    # Calculate average predictions
    avg_data = []
    for name in y_names:
        avg_prediction = predictions_sum[name] / predictions_count[name]
        avg_data.append({
            'FileName': name,
            'AveragePrediction': avg_prediction,
            'Label': y_true[y_names.index(name)]
        })

    data = pd.DataFrame(avg_data)
    data['Predicted'] = data['AveragePrediction'].apply(lambda x: int(x > 0.5))
    data['Score'] = data['AveragePrediction'] * 100

    # Compute overall accuracy
    accuracy = accuracy_score(data['Label'], data['Predicted'])
    accuracy = int(np.round((accuracy * 100)))

    # Separate negatives and positives
    negatives = data[data['Label'] == 0].sort_values('FileName')
    positives = data[data['Label'] == 1].sort_values('FileName')

    # Calculate accuracies for negatives and positives
    accuracy_negatives = len(negatives[negatives['Predicted'] == 0]) / len(negatives) * 100
    accuracy_positives = len(positives[positives['Predicted'] == 1]) / len(positives) * 100

    # Create subplots for visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'Accuracy-{model_name}: {accuracy}%', size=14)

    # Plot negatives
    axes[0].bar(negatives['FileName'], negatives['Score'], color=negatives['Predicted'].apply(lambda x: 'g' if x == 0 else 'r'))
    axes[0].set_ylim([0, 100])
    axes[0].axhline(50, c='black', linestyle='dotted', label='CNN_Models Threshold')
    axes[0].set_title(f'Negatives: {int(np.round(accuracy_negatives))}%')
    axes[0].set_ylabel('CNN_Models %')
    axes[0].tick_params(axis='x', rotation=90)

    # Create custom legend
    legend_handles = [mpatches.Patch(color='g', label='Predicted Correctly'),
                      mpatches.Patch(color='r', label='Predicted Incorrect'),
                      mpatches.Patch(color='black', label='CNN_Models Threshold', linestyle='dotted')]
    axes[0].legend(loc='upper left', handles=legend_handles)

    # Plot positives
    axes[1].bar(positives['FileName'], positives['Score'], color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[1].set_ylim([0, 100])
    axes[1].axhline(50, c='black', linestyle='dotted')
    axes[1].set_ylabel('CNN_Models %')
    axes[1].set_title(f'Positives: {int(np.round(accuracy_positives))}%')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout(pad=1)

    # Save or display the plot
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/{model_name}.png')
        plt.close(fig)
    else:
        plt.show()

# Function to use model text file to get parameter info
def load_model_text_file(model_path):
    text_path_base = model_path.split('.')[0]
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





