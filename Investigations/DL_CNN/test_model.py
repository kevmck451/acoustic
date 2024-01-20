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

    # Test accuracy of Models
    y_true = []
    y_pred = []
    y_pred_scores = []
    y_names = []

    model_path = str(model_path)
    model_info = load_model_text_file(model_path)
    path_model = Path(model_path)
    model_name = path_model.stem

    # LOAD DATA ----------------------------------------------------------------------
    print('Loading Features')
    features, labels = load_features(audio_path, model_info.get('Sample Length'), model_info.get('Sample Rate'),
                                     model_info.get('Multi Channel'), chunk_type, model_info.get('Process Applied'),
                                     model_info.get('Feature Type').lower(), model_info.get('Feature Parameters'))

    # PREDICT ------------------------------------------------------------------------
    print('Testing Model')
    model = load_model(model_path)

    text_file_name = model_name.split('_')[:-2]
    text_file_name = '_'.join(text_file_name)
    text_file_name = f'{text_file_name}/testing_features_files.txt'
    names_path = f'{Path(model_path).parent}/{text_file_name}'

    with open(names_path, 'r') as file:
        names = file.readlines()

    for i, (feature, label, name) in enumerate(zip(features, labels, names)):
        feature = np.expand_dims(feature, axis=0)
        y_new_pred = model.predict(feature)
        y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction
        y_names.append(name)

        # Retrieve true label
        y_true_class = label

        # Skip this file if it's not in our truth dictionary
        if y_true_class is None:
            print('Truth Not Found')
            continue

        # Append to our lists
        y_true.append(y_true_class)
        y_pred.append(y_pred_class)

        percent = np.round((y_new_pred[0][0] * 100), 2)
        y_pred_scores.append(percent)

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    accuracy = int(np.round((accuracy * 100)))
    # print(f'Accuracy: {accuracy}%')
    # print(f'Scores: {y_pred_scores}')

    # Create DataFrame
    data = pd.DataFrame({
        'FileName': y_names,
        'Label': y_true,
        'Predicted': y_pred,
        'Score': y_pred_scores
    })

    # Separate negatives and positives
    negatives = data[data['Label'] == 0].sort_values('Score')
    positives = data[data['Label'] == 1].sort_values('Score')

    # Calculate accuracies
    accuracy_negatives = len(negatives[negatives['Predicted'] == 0]) / len(negatives) * 100
    accuracy_positives = len(positives[positives['Predicted'] == 1]) / len(positives) * 100

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'Accuracy-{model_name}: {accuracy}%', size=14)

    # Plot negatives
    axes[0].bar(negatives['FileName'], negatives['Score'], color=negatives['Predicted'].apply(lambda x: 'g' if x == 0 else 'r'))
    axes[0].set_ylim([0, 100])  # Set y-axis limits for percentage
    axes[0].axhline(50, c='black', linestyle='dotted', label='CNN_Models Threshold')
    axes[0].set_title(f'Negatives: {int(np.round(accuracy_negatives))}%')
    axes[0].set_ylabel('CNN_Models %')
    axes[0].tick_params(axis='x', rotation=90)

    # Create custom legend handles and labels
    legend_handles = [
        mpatches.Patch(color='g', label='Predicted Correctly'),
        mpatches.Patch(color='r', label='Predicted Incorrect'),
        mpatches.Patch(color='black', label='CNN_Models Threshold', linestyle='dotted')]

    axes[0].legend(loc='upper left', handles=legend_handles)

    # Plot positives
    axes[1].bar(positives['FileName'], positives['Score'], color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[1].set_ylim([0, 100])  # Set y-axis limits for percentage
    axes[1].axhline(50, c='black', linestyle='dotted')
    axes[1].set_ylabel('CNN_Models %')
    axes[1].set_title(f'Positives: {int(np.round(accuracy_positives))}%')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout(pad=1)  # Adjust subplot parameters to give specified padding

    # audio_base = Audio_Abstract(filepath=audio_path)
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/{model_name}.png')
        plt.close(fig)
    else:
        plt.show()

    # return accuracy, y_pred_scores


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



if __name__ == '__main__':

    # testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Models Data/Static Flight_Analysis/Test 1'
    # testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Models Data/Static Detection/dataset'
    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Combinations'
    sample_lengths = [10, 8, 6, 4, 2]

    feature_types = ['spectral', 'filter1', 'mfcc']

    # model = load_model('CNN_Models/model_library/basic_1_mfcc_10_79_0.h5')
    # test_model_accuracy(model, testing_path, sample_lengths[0], feature_types[2], display=True)

    # model = load_model('CNN_Models/model_library/deep_1_mfcc_6_73_0.h5')
    # test_model_accuracy(model, testing_path, sample_lengths[2], feature_types[2], display=True)

    model = 'spectral_10_basic_1_100_1'
    sample_length = int(model.split('_')[1])
    feature_type = model.split('_')[0]

    model = load_model(f'Prediction/model_library/{model}.h5')
    test_model_accuracy(model, testing_path, sample_length, feature_type, display=True)




