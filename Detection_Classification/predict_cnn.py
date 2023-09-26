# File to load audio and make predictions based on which model is chosen and display those predictions


from Acoustic import process
from Acoustic import audio_abstract


from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm as progress_bar




def make_prediction(model_path, audio_path, display):

    # LOAD DATA ------------------------------------------------------------------------
    print('Loading Mission Audio')
    audio = audio_abstract.Audio_Abstract(filepath=audio_path)
    # print(audio)

    # Determine sample length
    path_model = Path(model_path)
    model_name = path_model.stem
    sample_length = int(model_name.split('_')[3])

    chunks_list = process.generate_windowed_chunks(audio, window_size=sample_length)

    # EXTRACT ------------------------------------------------------------------------
    print('Extracting Features')
    feature_type = model_name.split('_')[2]
    features_list = []
    for audio in progress_bar(chunks_list):
        if feature_type == 'spectral':
            feature = process.spectrogram(audio)
        elif feature_type == 'filter1':
            feature = process.custom_filter_1(audio)
        elif feature_type == 'mfcc':
            feature = process.mfcc(audio)
        elif feature_type == 'zcr':
            feature = process.zcr(audio)

        features_list.append(feature)  # Add Feature

    features_list = np.array(features_list)
    # features_list = np.squeeze(features_list, axis=1)
    # features_list = features_list[..., np.newaxis]



    # PREDICT ------------------------------------------------------------------------
    print('Making Predictions')
    model = load_model(model_path)

    predictions = []
    for feature in features_list:
        y_new_pred = model.predict(feature)
        percent = np.round((y_new_pred[0][0] * 100), 2)
        predictions.append(percent)

    time = list(range(0, len(predictions), 1))


    if display:
        fig, axs = plt.subplots(1, 1, figsize=(12, 4))
        plt.suptitle(f'Engine Classification Model: {model_name}')
        bar_colors = ['r' if value >= 50 else 'g' for value in predictions]
        axs.bar(time, predictions, width=int(sample_length/sample_length), color=bar_colors)
        axs.set_title(f'Predictions of {audio.path.stem}')
        axs.set_xlabel('Time')
        axs.set_ylabel('Predictions')
        axs.set_ylim((0, 100))
        axs.axhline(50, c='black', linestyle='dotted')

        # Add custom legend for colors
        red_patch = mpatches.Patch(color='red', label='Threat Detected')
        green_patch = mpatches.Patch(color='green', label='No Threat')
        axs.legend(handles=[red_patch, green_patch], loc='upper right')

        plt.tight_layout(pad=1)
        plt.show()



if __name__ == '__main__':
    model_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Detection_Classification/Engine_Classification/Prediction/model_library/basic_1_mfcc_6_99_0.h5'

    # Experiment 1
    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests'
    audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 1.wav'
    # audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 2.wav'
    # audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 3.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 1.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 2.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 3.wav'

    # make_prediction(model_path, audio_path, display=True)

    # Experiment 2
    base_path_2 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests'
    audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 1.wav'
    # audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 2.wav'
    # audio_path = f'{base_path_1}/Campus/Campus Construction/Construction 3.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 1.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 2.wav'
    # audio_path = f'{base_path_1}/Orlando 23/Samples/Ambient/Orlando Ambient 3.wav'

    make_prediction(model_path, audio_path, display=True)


