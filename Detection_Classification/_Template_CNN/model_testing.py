
from Investigations.DL_CNN.test_model import test_model_accuracy

from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


def run_test_set(model_path, test_path, **kwargs):

    chunk_type = ['regular', 'window']
    save_path = f'{save_directory}/{Path(model_path).stem}'
    # Path(save_path).mkdir(exist_ok=True, parents=True)

    path_model = Path(model_path)
    model_name = path_model.stem

    names_list, predict_scores, labels_list = test_model_accuracy(model_path, test_path, chunk_type[0])

    # print(names_list)

    height = [name.split('_')[0] for name in names_list] # height above target
    sample_type = [name.split('_')[1] for name in names_list] # target character
    microphone = [name.split('_')[2].strip() for name in names_list]
    samp_mic = [f"{name.split('_')[1]}_{name.split('_')[2].strip()}" for name in names_list]
    data = pd.DataFrame({
        'FileName': names_list,
        'Prediction': predict_scores,
        'Label': labels_list,
        'Height': height,
        'SampleType': sample_type,
        'Mic': microphone,
        'SampMic': samp_mic, })

    data['Predicted'] = data['Prediction'].apply(lambda x: int(x > 0.5))
    data['Score'] = data['Prediction'] * 100

    # Compute overall accuracy
    accuracy = accuracy_score(data['Label'], data['Predicted'])
    accuracy = int(np.round((accuracy * 100)))

    # Separate negatives and positives
    negatives = data[data['Label'] == 0].sort_values('FileName')
    positives = data[data['Label'] == 1].sort_values('FileName')
    m10_predictions = data[(data['Height'] == '6m') & data['Label'] == 1].sort_values('SampleType')
    m20_predictions = data[(data['Height'] == '18m') & data['Label'] == 1].sort_values('SampleType')
    m30_predictions = data[(data['Height'] == '26m') & data['Label'] == 1].sort_values('SampleType')
    m40_predictions = data[(data['Height'] == '38m') & data['Label'] == 1].sort_values('SampleType')

    # Reset the index for each subset DataFrame and create the 'UniSampMic' column
    m10_predictions = m10_predictions.reset_index(drop=True)
    m10_predictions['UniSampMic'] = m10_predictions['SampMic'] + '_' + (m10_predictions.index + 1).astype(str)

    m20_predictions = m20_predictions.reset_index(drop=True)
    m20_predictions['UniSampMic'] = m20_predictions['SampMic'] + '_' + (m20_predictions.index + 1).astype(str)

    m30_predictions = m30_predictions.reset_index(drop=True)
    m30_predictions['UniSampMic'] = m30_predictions['SampMic'] + '_' + (m30_predictions.index + 1).astype(str)

    m40_predictions = m40_predictions.reset_index(drop=True)
    m40_predictions['UniSampMic'] = m40_predictions['SampMic'] + '_' + (m40_predictions.index + 1).astype(str)

    # Calculate accuracies for negatives and positives
    accuracy_negatives = int(len(negatives[negatives['Predicted'] == 0]) / len(negatives) * 100)
    accuracy_positives = int(len(positives[positives['Predicted'] == 1]) / len(positives) * 100)
    accuracy_10m = int(len(m10_predictions[m10_predictions['Predicted'] == 1]) / len(m10_predictions) * 100)
    accuracy_20m = int(len(m20_predictions[m20_predictions['Predicted'] == 1]) / len(m20_predictions) * 100)
    accuracy_30m = int(len(m30_predictions[m30_predictions['Predicted'] == 1]) / len(m30_predictions) * 100)
    accuracy_40m = int(len(m40_predictions[m40_predictions['Predicted'] == 1]) / len(m40_predictions) * 100)

    # Create subplots for visualization ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(22, 14))
    fig.suptitle(
        f'Model: {model_name} | Test 1 Accuracy: {accuracy}%\nDetector: Every Sample | Negatives Accuracy: {accuracy_negatives}%',
        size=12)

    # Create custom legend
    legend_handles = [mpatches.Patch(color='g', label='Predicted Correctly'),
                      mpatches.Patch(color='r', label='Predicted Incorrect'), ]
    # mpatches.Patch(color='black', label='CNN_Models Threshold', linestyle='dotted')

    label_size = 8
    # 10m ------------------------------------
    axes[0].bar(m10_predictions['UniSampMic'], m10_predictions['Score'],
                color=m10_predictions['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[0].set_ylim([0, 100])
    axes[0].axhline(50, c='black', linestyle='dotted', label='CNN_Models Threshold')
    axes[0].set_title(f'6m Samples: {int(np.round(accuracy_10m))}%')
    axes[0].set_ylabel('Confidence %')
    axes[0].set_xticks(np.arange(len(m10_predictions['UniSampMic'])))
    axes[0].set_xticklabels(m10_predictions['UniSampMic'])
    axes[0].tick_params(axis='x', rotation=90, labelsize=label_size)

    # Set the color of each tick label based on SampleType
    for label, sample_type in zip(axes[0].get_xticklabels(), m10_predictions['SampleType']):
        if sample_type == 't':
            label.set_color('black')

    axes[0].legend(loc='upper left', handles=legend_handles)

    # 20m ------------------------------------
    axes[1].bar(m20_predictions['UniSampMic'], m20_predictions['Score'],
                color=m20_predictions['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[1].set_ylim([0, 100])
    axes[1].axhline(50, c='black', linestyle='dotted')
    axes[1].set_ylabel('Confidence %')
    axes[1].set_title(f'18m Samples: {int(np.round(accuracy_20m))}%')
    axes[1].set_xticks(np.arange(len(m20_predictions['UniSampMic'])))
    axes[1].set_xticklabels(m20_predictions['UniSampMic'])
    axes[1].tick_params(axis='x', rotation=90, labelsize=label_size)

    # Set the color of each tick label based on SampleType
    for label, sample_type in zip(axes[1].get_xticklabels(), m20_predictions['SampleType']):
        if sample_type == 't':
            label.set_color('black')


    # 30m ------------------------------------
    axes[2].bar(m30_predictions['UniSampMic'], m30_predictions['Score'],
                color=m30_predictions['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[2].set_ylim([0, 100])
    axes[2].axhline(50, c='black', linestyle='dotted')
    axes[2].set_ylabel('Confidence %')
    axes[2].set_title(f'26m Samples: {int(np.round(accuracy_30m))}%')
    axes[2].set_xticks(np.arange(len(m30_predictions['UniSampMic'])))
    axes[2].set_xticklabels(m30_predictions['UniSampMic'])
    axes[2].tick_params(axis='x', rotation=90, labelsize=label_size)

    # Set the color of each tick label based on SampleType
    for label, sample_type in zip(axes[2].get_xticklabels(), m30_predictions['SampleType']):
        if sample_type == 't':
            label.set_color('black')


    # 40m ------------------------------------
    axes[3].bar(m40_predictions['UniSampMic'], m40_predictions['Score'],
                color=m40_predictions['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[3].set_ylim([0, 100])
    axes[3].axhline(50, c='black', linestyle='dotted')
    axes[3].set_ylabel('Confidence %')
    axes[3].set_title(f'38m Samples: {int(np.round(accuracy_40m))}%')
    axes[3].set_xticks(np.arange(len(m40_predictions['UniSampMic'])))
    axes[3].set_xticklabels(m40_predictions['UniSampMic'])
    axes[3].tick_params(axis='x', rotation=90, labelsize=label_size)

    # Set the color of each tick label based on SampleType
    for label, sample_type in zip(axes[3].get_xticklabels(), m40_predictions['SampleType']):
        if sample_type == 't':
            label.set_color('black')


    # ----------------------------------
    plt.tight_layout(pad=2)

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
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'

    # Make sure to change TEMPLATE for your current project
    experiment_directory_name = 'Angel_Hover'
    test_directory_name = 'test 1'
    save_directory_name = 'Angel Hover'

    model_path = Path(f'{base_path}/Detection_Classification/{experiment_directory_name}/model_library')
    test_directory_path = f'{test_base_path}/ML Model Data/Angel_Hover/{test_directory_name}'
    save_directory = f'{save_base_dir}/{save_directory_name}'

    index_num = 0
    layer_num = 3
    length = 4


    for model in model_path.iterdir():
        if 'h5' in str(model):
            if int(str(model).split('_')[-3][0]) == length:
                if int(str(model).split('_')[-1].split('.')[0]) == index_num:
                    if int(str(model).split('_')[-2].split('-')[0]) == layer_num:
                        run_test_set(model.resolve(), test_directory_path, save=True, save_path=save_directory)