
from Investigations.DL_CNN.test_model import test_model_accuracy

from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np



# Detector for full sample averages
def detector_sample_average(model_path, test_path, **kwargs):
    '''
    Detector that takes full sample and chops it up into whatever
    length the model was trained on and then averages the predictions
    for each sample
    '''

    path_model = Path(model_path)
    model_name = path_model.stem

    chunk_type = ['regular', 'window']

    names_list, predict_scores, labels_list = test_model_accuracy(model_path, test_path, chunk_type[0])

    predictions_sum = {}
    predictions_count = {}
    y_true = []
    y_names = []

    for name, score, label in zip(names_list, predict_scores, labels_list):

        # Aggregate predictions
        if name not in predictions_sum:
            predictions_sum[name] = 0
            predictions_count[name] = 0
        predictions_sum[name] += score
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

# Detector for full sample averages with windows
def detector_sample_average_windowed(model_path, test_path, **kwargs):
    '''
        Detector that takes full sample and chops it up into whatever
        length the model was trained on and then averages the predictions
        for each sample
        '''

    path_model = Path(model_path)
    model_name = path_model.stem

    chunk_type = ['regular', 'window']

    names_list, predict_scores, labels_list = test_model_accuracy(model_path, test_path, chunk_type[1])

    predictions_sum = {}
    predictions_count = {}
    y_true = []
    y_names = []

    for name, score, label in zip(names_list, predict_scores, labels_list):

        # Aggregate predictions
        if name not in predictions_sum:
            predictions_sum[name] = 0
            predictions_count[name] = 0
        predictions_sum[name] += score
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
    axes[0].bar(negatives['FileName'], negatives['Score'],
                color=negatives['Predicted'].apply(lambda x: 'g' if x == 0 else 'r'))
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
    axes[1].bar(positives['FileName'], positives['Score'],
                color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
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

# shows every sample's individual prediction
def detector_every_sample(model_path, test_path, **kwargs):
    '''
    Detector that takes full sample and chops it up into whatever
    length the model was trained on and then averages the predictions
    for each sample
    '''

    path_model = Path(model_path)
    model_name = path_model.stem

    chunk_type = ['regular', 'window']

    names_list, predict_scores, labels_list = test_model_accuracy(model_path, test_path, chunk_type[0])

    data = pd.DataFrame({'FileName': names_list, 'Prediction': predict_scores, 'Label': labels_list})

    data['Predicted'] = data['Prediction'].apply(lambda x: int(x > 0.5))
    data['Score'] = data['Prediction'] * 100

    # Compute overall accuracy
    accuracy = accuracy_score(data['Label'], data['Predicted'])
    accuracy = int(np.round((accuracy * 100)))

    # Separate negatives and positives
    negatives = data[data['Label'] == 0].sort_values('FileName').copy()
    positives = data[data['Label'] == 1].sort_values('FileName').copy()

    negatives['UniqueFileName'] = negatives['FileName'] + '_' + negatives.index.astype(str)
    positives['UniqueFileName'] = positives['FileName'] + '_' + positives.index.astype(str)

    # Calculate accuracies for negatives and positives
    accuracy_negatives = len(negatives[negatives['Predicted'] == 0]) / len(negatives) * 100
    accuracy_positives = len(positives[positives['Predicted'] == 1]) / len(positives) * 100

    # Create subplots for visualization
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle(f'Accuracy-{model_name}: {accuracy}%', size=14)

    # Plot negatives
    axes[0].bar(negatives['UniqueFileName'], negatives['Score'], color=negatives['Predicted'].apply(lambda x: 'g' if x == 0 else 'r'))
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
    axes[1].bar(positives['UniqueFileName'], positives['Score'], color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
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










if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    test_base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    test_directory_path = f'{test_base_path}/ML Model Data/Static Test 3/testing 1'
    model = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/' \
            'Detection_Classification/_Template_CNN/model_library/mfcc_13_2s_4-layers_0.h5'

    detector_sample_average(model, test_directory_path)