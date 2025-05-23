# File to test the accuracy of a model against some known samples that were excluded from test data


from Detection_Classification.CNN_Models.load_data import load_audio_data
from Detection_Classification.CNN_Models.dataset_info import *

from sklearn.metrics import accuracy_score
from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches



# Function to test ML Models's Accuracy
def test_model_accuracy(model, directory, sample_length, feature_type, display=False, stats=False):
    if stats:
        # Get the model's architecture
        model.summary()

        # Get the optimizer configuration
        optimizer_config = model.optimizer.get_config()
        print("Optimizer Configuration:", optimizer_config)

    Test_Directory = Path(str(directory))

    # Test accuracy of Models
    y_true = []
    y_pred = []
    y_pred_scores = []
    y_names = []

    features, labels = load_audio_data(Test_Directory, sample_length, feature_type)

    for i, (feature, label) in enumerate(zip(features, labels)):
        feature = np.expand_dims(feature, axis=0)
        y_new_pred = model.predict(feature)
        y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction
        y_names.append(f'Sample {i}')


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
    print(f'Accuracy: {accuracy}%')
    print(f'Scores: {y_pred_scores}')

    if display:
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
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Spectral_Model Accuracy-{Test_Directory.stem}: {accuracy}%', size=14)

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
        plt.show()

    return accuracy, y_pred_scores

if __name__ == '__main__':
    # Models List
    model = load_model('Prediction/model_library/mfcc_6_basic_1_99_0.h5')
    # model = load_model('CNN_Models/model_library/basic_1_spectral_6_85_0.h5')

    testing_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Classification/tests'

    sample_lengths = [10, 8, 6, 4, 2]

    feature_types = ['spectral', 'filter1', 'mfcc']

    test_model_accuracy(model, testing_path, sample_lengths[2], feature_types[2], display=True)


