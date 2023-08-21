
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

from tqdm import tqdm as progress_bar
from pathlib import Path
import numpy as np



# Load Data from a Dataset with Labels and Extract Features
def load_audio_data(path, length, feature_type):
    print('Loading Dataset')
    audio_ob_list = []
    label_list = []
    for file in progress_bar(Path(path).rglob('*.wav')):
        audio = Audio_Abstract(filepath=file)
        if audio.num_channels == 1:
            chunks_list, labels = process.generate_chunks(audio, length=length, training=True)
            # chunks_list, labels = process.generate_windowed_chunks(audio, window_size=length, training=True)
            audio_ob_list.extend(chunks_list)  # Flattening the chunks_list
            label_list.extend(labels)  # Flattening the labels
        else: # it's 4 channel
            channel_list = process.channel_to_objects(audio)
            for channel in channel_list:
                chunks_list, labels = process.generate_chunks(channel, length=length, training=True)
                # chunks_list, labels = process.generate_windowed_chunks(audio, window_size=length, training=True)
                audio_ob_list.extend(chunks_list)

                label_list.extend(labels)  # Flattening the labels

    master_ob_list = list(audio_ob_list)  # Creating a new 1D list
    master_label_list = list(label_list)  # Creating a new 1D list

    print('Preprocessing Data')
    prepro_ob_list = []
    # for audio in progress_bar(master_ob_list):
    #     # Spectral Subtraction
    #     process.spectra_subtraction_hex(audio)


    print('Extracting Features')
    features_list = []
    for audio in progress_bar(master_ob_list):
        if feature_type == 'spectral':
            feature = process.spectrogram(audio)
        elif feature_type == 'filter1':
            feature = process.custom_filter_1(audio)
        elif feature_type == 'mfcc':
            feature = process.mfcc(audio)
        elif feature_type == 'zcr':
            feature = process.zcr(audio)


        features_list.append(feature)  # Add Feature
        # print(feature.shape)
        # print(feature.dtype)

    # make into array and reshape to (6788, 1310, 188, 1) - (samples, freq, reduced time, batch)
    features_list = np.array(features_list)
    features_list = np.squeeze(features_list, axis=1)
    features_list = features_list[..., np.newaxis]

    return features_list, np.array(master_label_list)




if __name__ == '__main__':
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    sample_length = 10
    # sample_length = 5
    # sample_length = 2

    feature_type = 'spectral'
    # feature_type = 'filter1'
    # feature_type = 'mfcc'


    features, labels = load_audio_data(dataset, sample_length, feature_type)



