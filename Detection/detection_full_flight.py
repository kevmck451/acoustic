# File for starting point to detect targets from dataset


from Detection.models.dataset_info import *
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

from keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import statistics


def full_flight_detection(filepath, model_path):

    # LOAD DATA ------------------------------------------------------------------------
    print('Loading Mission Audio')
    audio = Audio_Abstract(filepath=filepath)
    channel_list = process.channel_to_objects(audio)

    audio_ob_list = []
    for channel in channel_list:
        chunks_list, labels = process.generate_chunks(channel, length=2)
        audio_ob_list.append(chunks_list)

    # EXTRACT ------------------------------------------------------------------------
    print('Extracting Features')
    features_list = []
    for channel in audio_ob_list:
        feature_split_list = []
        for audio in channel:
            feature = process.spectrogram(audio)
            feature_split_list.append(feature)  # Add Feature
        features_list.append(np.array(feature_split_list))

    # PREDICT ------------------------------------------------------------------------
    print('Making Predictions')
    model_dir = model_path
    model = load_model(model_dir)

    predictions_list = []

    for channel in features_list:
        predictions = []
        for feature in channel:
            y_new_pred = model.predict(feature)
            y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction
            percent = np.round((y_new_pred[0][0] * 100), 2)
            predictions.append(percent)

        predictions_list.append(predictions)

    time = list(range(0, (len(predictions_list[0]) * 2), 2))

    # AVERAGE CHANNEL PREDICTIONS ------------------------------------------------------------------------
    # print(time)
    # print(predictions_list)

    averaged_predictions = [statistics.mean(values) for values in zip(*predictions_list)]

    print(averaged_predictions)

    fig, axs = plt.subplots(5, 1, figsize=(14, 8))
    plt.suptitle(f'Sound Source Detection-Model: {Path(model_dir).stem}')

    # Loop over your 4 lists
    for i in range(4):
        bar_colors = ['g' if value >= 50 else 'r' for value in predictions_list[i]]
        axs[i].bar(time, predictions_list[i], color=bar_colors)
        axs[i].set_title(f"Channel {i + 1}")
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Predictions')
        axs[i].set_ylim((0,100))
        axs[i].axhline(50, c='black', linestyle='dotted')

    # Plot averaged_predictions
    bar_colors = ['g' if value >= 50 else 'r' for value in averaged_predictions]
    axs[4].bar(time, averaged_predictions, color=bar_colors)
    axs[4].set_title(f"Averaged Predictions")
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Predictions')
    axs[4].set_ylim((0, 100))
    axs[4].axhline(50, c='black', linestyle='dotted')

    # Ensure the subplots do not overlap
    plt.tight_layout(pad=1)
    plt.show()


if __name__ == '__main__':
    mission_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Agricenter/Hex Flight 7/Hex 7.wav'
    # mission_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Campus/Static Tests/Static Test 2/RAW/4.wav'
    model_path = 'models/Spectral_Model_2s/model_library/detect_spec_2_96_0.h5'
    full_flight_detection(mission_path, model_path)

