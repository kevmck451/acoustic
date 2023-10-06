# File to load audio and make predictions based on which model is chosen and display those predictions
# Can only load mono files currently

from Acoustic import process
from Acoustic import audio_abstract


from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm as progress_bar
import tkinter as tk
from tkinter import filedialog


def make_prediction(model_path, audio_path, **kwargs):

    # LOAD DATA ------------------------------------------------------------------------
    print('Loading Mission Audio')
    audio = audio_abstract.Audio_Abstract(filepath=audio_path)
    # print(audio)

    # Determine sample length
    path_model = Path(model_path)
    model_name = path_model.stem
    sample_length = int(model_name.split('_')[1])

    chunks_list = process.generate_windowed_chunks(audio, window_size=sample_length)

    # EXTRACT ------------------------------------------------------------------------
    print('Extracting Features')
    feature_type = model_name.split('_')[0]
    features_list = []
    for audio in progress_bar(chunks_list):
        if feature_type == 'spectral':
            feature_params = kwargs.get('feature_params', (70, 4000))
            feature = process.spectrogram(audio, feature_params=feature_params)
        elif feature_type == 'filter1':
            feature = process.custom_filter_1(audio)
        elif feature_type == 'mfcc':
            feature_params = kwargs.get('feature_params', (70, 4000))
            feature = process.mfcc(audio, feature_params=feature_params)
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

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    plt.suptitle(f'Ambient vs Engine Model: {model_name}')
    bar_colors = ['r' if value >= 50 else 'g' for value in predictions]
    axs.bar(time, predictions, width=int(sample_length/sample_length), color=bar_colors)
    axs.set_title(f'Predictions of {audio.name}')
    axs.set_xlabel('Time')
    axs.set_ylabel('Predictions')
    axs.set_ylim((0, 100))
    axs.axhline(50, c='black', linestyle='dotted')

    postive = kwargs.get('positive_label', 'Threat Detected')
    negative = kwargs.get('negative_label', 'No Threat')
    # Add custom legend for colors
    red_patch = mpatches.Patch(color='red', label=postive)
    green_patch = mpatches.Patch(color='green', label=negative)
    axs.legend(handles=[red_patch, green_patch], loc='upper right')

    plt.tight_layout(pad=1)

    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', str(audio.path))
    if save:
        plt.savefig(f'{save_path}/{audio.name}.png')
    else:
        plt.show()

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path

if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Detection_Classification'
    # model_path = f'{base_path}/Engine_Classification/Prediction/model_library/mfcc_6_basic_1_99_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_6_basic_1_94_0.h5'
    model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_6_deep_3_100_0.h5'

# Experiment 1 -------------------------------------------------------------
    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'

    directory_list = [
                    f'{base_path_1}/Field Tests/Campus/Construction 2',
                    f'{base_path_1}/Field Tests/Random',
                    f'{base_path_1}/Isolated Samples/Testing',
                    f'{base_path_1}/Field Tests/Orlando 23/Samples/Ambient',
                    f'{base_path_1}/Field Tests/Campus/Generator/',
                    # f'{base_path_1}/Combinations/Ambient Diesel'
                   ]

    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis/'
    save_directory = f'{save_base_dir}/Engine vs Ambient/MFCC/Model 4'
    for path in directory_list:
        for audio_path in Path(path).iterdir():
            print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, save=True, save_path=save_directory,
                                positive_label='Engine Detected', negative_label='Ambient Noise')



    # Experiment X -------------------------------------------------------------
    # audio_path = select_file()
    # print(audio_path)
    # make_prediction(model_path, audio_path, display=True)

    # audio_path = select_file()
    # audio_dir = Path(audio_path).parent
    # # print(audio_dir)
    # # print(audio_path)
    #
    # for path in audio_dir.iterdir():
    #     if 'wav' in path.suffix:
    #         make_prediction(model_path, path, display=True)

