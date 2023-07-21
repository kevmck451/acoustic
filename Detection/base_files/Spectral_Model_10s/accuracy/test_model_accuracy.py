# File to test the accuracy of a model against some known samples that were excluded from test data


from Detection.base_files.feature_ex_template import extract_features
from Detection.base_files.Spectral_Model_10s.accuracy.generate_truth import generate_truth
from Prediction.dataset_info import *

from sklearn.metrics import accuracy_score
from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches



# Function to test ML Model's Accuracy
def test_model_accuracy(model, directory, truth, display=False, stats=False):
    if stats:
        # Get the model's architecture
        model.summary()

        # Get the optimizer configuration
        optimizer_config = model.optimizer.get_config()
        print("Optimizer Configuration:", optimizer_config)

    Test_Directory = Path(directory)

    # Test accuracy of Model


    y_true = []
    y_pred = []
    y_pred_scores = []
    y_names = []

    for file in Test_Directory.rglob('*.wav'):
        y_names.append(file.stem)

        # Load and preprocess the new audio sample
        feature = extract_features(file, 10)
        feature = np.array([feature])
        feature = feature[..., np.newaxis]

        # Predict class
        print(feature.shape)
        y_new_pred = model.predict(feature)
        y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction
        # y_pred_score = y_new_pred[0][0]  # Store the prediction score

        # Retrieve true label
        y_true_class = truth.get(file.stem, None)

        # Skip this file if it's not in our truth dictionary
        if y_true_class is None:
            print('Truth Not Found')
            continue

        # Append to our lists
        y_true.append(y_true_class)
        y_pred.append(y_pred_class)

        percent = np.round((y_new_pred[0][0] * 100), 2)
        y_pred_scores.append(percent)
        # print(f'File: {file.stem} / Percent: {percent}%')

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    accuracy = int(np.round((accuracy * 100)))
    print(f'Accuracy: {accuracy}%')

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
        axes[0].axhline(50, c='black', linestyle='dotted', label='Prediction Threshold')
        axes[0].set_title(f'Negatives: {int(np.round(accuracy_negatives))}%')
        axes[0].set_ylabel('Prediction %')
        axes[0].tick_params(axis='x', rotation=90)

        # Create custom legend handles and labels
        legend_handles = [
            mpatches.Patch(color='g', label='Predicted Correctly'),
            mpatches.Patch(color='r', label='Predicted Incorrect'),
            mpatches.Patch(color='black', label='Prediction Threshold', linestyle='dotted')]

        axes[0].legend(loc='upper left', handles=legend_handles)

        # Plot positives
        axes[1].bar(positives['FileName'], positives['Score'], color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
        axes[1].set_ylim([0, 100])  # Set y-axis limits for percentage
        axes[1].axhline(50, c='black', linestyle='dotted')
        axes[1].set_ylabel('Prediction %')
        axes[1].set_title(f'Positives: {int(np.round(accuracy_positives))}%')
        axes[1].tick_params(axis='x', rotation=90)

        plt.tight_layout(pad=1)  # Adjust subplot parameters to give specified padding
        plt.show()

    return accuracy, y_names, y_pred_scores

if __name__ == '__main__':

    # Model List
    # model = load_model('models/Spectral_Detection_Model.h5')
    model = load_model('../../../../Prediction/model_library/detect_spec_10_100_0.h5')

    truth = generate_truth(directory_test_1)
    test_model_accuracy(model, directory_test_1, truth)


