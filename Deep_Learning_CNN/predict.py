# File to load audio and make predictions based on which model is chosen and display those predictions
# Can only load mono files currently

from Acoustic import process
from Acoustic.audio_abstract import Audio_Abstract
from Deep_Learning_CNN.load_features import load_audio_generator
from Deep_Learning_CNN.load_features import preprocess_files
from Deep_Learning_CNN.load_features import extract_feature
from Deep_Learning_CNN.load_features import format_features


from keras.models import load_model
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import filedialog


# Function to make prediction and display it
def make_prediction(model_path, audio_path, **kwargs):
    audio_base = Audio_Abstract(filepath=audio_path)

    model_info = load_model_text_file(model_path)
    path_model = Path(model_path)
    model_name = path_model.stem

    # LOAD DATA ----------------------------------------------------------------------
    print('Loading Features')
    feature_list_master = []
    for audio_list, _ in load_audio_generator(audio_path,
                                         model_info.get('Sample Rate'),
                                         model_info.get('Sample Length'),
                                         model_info.get('Multi Channel')):
        for audio in audio_list:
            audio = preprocess_files(audio, model_info.get('Process Applied'))
            features = extract_feature(audio,
                                       model_info.get('Feature Type').lower(),
                                       model_info.get('Feature Parameters'))
            feature_list_master.append(features)

    features_list = format_features(feature_list_master)

    # PREDICT ------------------------------------------------------------------------
    print('Making Predictions')
    model = load_model(model_path)

    predictions = []
    for feature in features_list:
        feature = np.expand_dims(feature, axis=0)
        y_new_pred = model.predict(feature)
        percent = np.round((y_new_pred[0][0] * 100), 2)
        predictions.append(percent)

    time = list(range(0, len(predictions), 1))

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    plt.suptitle(f'Ambient vs Engine Model: {model_name}')
    bar_colors = ['r' if value >= 50 else 'g' for value in predictions]
    axs.bar(time, predictions, width=int(model_info.get('Sample Length')/model_info.get('Sample Length')), color=bar_colors)
    axs.set_title(f'Predictions of {audio_base.name}')
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
    save_path = kwargs.get('save_path', str(audio_base.path))
    if save:
        plt.savefig(f'{save_path}/{audio_base.name}.png')
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

# Function to use GUI to load file / directory
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path

if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
    # model_path = f'{base_path}/Engine_Classification/Prediction/model_library/mfcc_50_6_basic_1_99_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_50_6_basic_1_87_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_50_6_basic_1_94_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_50_6_deep_3_100_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_100_6_basic_1_85_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/mfcc_100_6_deep_3_83_0.h5'
    # model_path = f'{base_path}/Engine_Ambient/Prediction/model_library/spectral_70-2600_6_basic_1_24_0.h5'
    # model_path = f'{base_path}/Deep_Learning_CNN/model_library/spectral_70-10000_10s_4-layers_0.h5'
    model_path = f'{base_path}/Deep_Learning_CNN/model_library/spectral_70-6000_10s_4-layers_0.h5'
# Experiment 1 -------------------------------------------------------------
    save_base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
    save_directory_1 = f'{save_base_dir}/Engine vs Ambient/MFCC/test'
    save_directory_2 = f'{save_base_dir}/Engine vs Ambient/MFCC/test'

    base_path_1 = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'
    directory_list = [
                    f'{base_path_1}/Field Tests/Campus/Construction 2',
                    f'{base_path_1}/Field Tests/Random',
                    f'{base_path_1}/Isolated Samples/Testing',
                    f'{base_path_1}/Field Tests/Orlando 23/Samples/Ambient',
                    f'{base_path_1}/Field Tests/Campus/Generator/',
                   ]

    for path in directory_list:
        for audio_path in Path(path).iterdir():
            print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, save=True, save_path=save_directory_1,
                                positive_label='Engine Detected', negative_label='Ambient Noise')

    directory_list = [f'{base_path_1}/Combinations/Ambient Diesel']
    for path in directory_list:
        for audio_path in Path(path).iterdir():
            print(audio_path)
            if 'wav' in audio_path.suffix:
                make_prediction(model_path, audio_path, save=True, save_path=save_directory_2,
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

