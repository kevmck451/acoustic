
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

from tqdm import tqdm as progress_bar
from pathlib import Path
import numpy as np


def load_audio_data(path, length, feature_type, **kwargs):
    print('Loading Dataset')
    print('Preprocessing')
    print('Extracting Features')
    feature_params = kwargs.get('feature_params', 'None')
    spec_window_size = kwargs.get('spec_window_size', 'None')
    hop_size = kwargs.get('hop_length', 'None')

    features_list = []
    label_list = []

    for file in progress_bar(Path(path).rglob('*.wav')):
        audio = Audio_Abstract(filepath=file, sample_rate=20000)

        # Get list of audio chunks depending on channel count
        audio_ob_list = get_audio_chunks(audio, length)
        label_list.extend(audio_ob_list[1])

        # Preprocess and extract features
        for audio_chunk in audio_ob_list[0]:
            normalized_audio = process.normalize(audio_chunk)
            # Compression (Dynamic Range)
            # Noise Reduction
            feature = extract_feature(normalized_audio, feature_type, feature_params, spec_window_size, hop_size)
            features_list.append(feature)

    features_list = format_features(features_list)
    return features_list, np.array(label_list)

def get_audio_chunks(audio, length):
    chunks_labels_list = []
    if audio.num_channels == 1:
        chunks_labels_list.append(process.generate_chunks(audio, length=length, training=True))
    else:  # it's 4 channel
        channel_list = process.channel_to_objects(audio)
        for channel in channel_list:
            chunks_labels_list.append(process.generate_chunks(channel, length=length, training=True))
    # Flattening the list of chunks and labels
    audio_chunks, labels = zip(*chunks_labels_list)
    return [chunk for sublist in audio_chunks for chunk in sublist], [label for sublist in labels for label in sublist]

def extract_feature(audio, feature_type, feature_params='None', spec_window_size='None', hop_length='None'):
    if feature_type == 'spectral':
        return process.spectrogram(audio, feature_params=feature_params, window_size=spec_window_size, hop_length=hop_length)
    elif feature_type == 'mfcc':
        return process.mfcc(audio, feature_params=feature_params)
    elif feature_type == 'filter1':
        return process.custom_filter_1(audio)
    elif feature_type == 'zcr':
        return process.zcr(audio)
    else: raise Exception('Error with feature type')

def format_features(features_list):
    features_list = np.array(features_list)
    features_list = np.squeeze(features_list, axis=1)
    return features_list[..., np.newaxis]


if __name__ == '__main__':
    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    sample_length = 10
    # sample_length = 5
    # sample_length = 2

    feature_type = 'spectral'
    # feature_type = 'filter1'
    # feature_type = 'mfcc'


    features, labels = load_audio_data(dataset, sample_length, feature_type)



