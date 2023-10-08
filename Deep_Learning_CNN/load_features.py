# loading data for CNN models

from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process
from Acoustic.utils import time_class

from tqdm import tqdm as progress_bar
from pathlib import Path
import numpy as np


def load_features(filepath, length, sample_rate, multi_channel, process_list, feature_type, feature_params):

    if not Path(filepath).exists():
        raise Exception('Directory does not exists')
    if type(length) is not int:
        raise Exception('Length needs to be an integer')
    if length < 1:
        raise Exception('Length needs to be greater than 1')
    if type(sample_rate) is not int:
        raise Exception('Sample Rate needs to be an integer')
    if sample_rate < 10_000 or sample_rate > 48_0001:
        raise Exception('Sample Rate Out of Range')
    if type(feature_params.get('bandwidth')[0]) is not int or type(feature_params.get('bandwidth')[1]) is not int:
        raise Exception('Bandwidth must be integers')
    if feature_params.get('bandwidth')[0] < 50 or feature_params.get('bandwidth')[1] > int(sample_rate/2):
        raise Exception('Bandwidth is out of range')
    if feature_params.get('window_size') % 2 != 0:
        raise Exception('Window Size needs to be a power of 2')

    feat = 'None'
    if feature_type == 'spectral':
        bandwidth = feature_params.get('bandwidth')
        window = feature_params.get('window_size')
        feat = f'{bandwidth[0]}-{bandwidth[1]}-{window}'
    if feature_type == 'mfcc':
        feat = feature_params.get('n_coeffs')

    feature_label_dir_path = Path(f'{Path.cwd()}/features_labels')
    feature_label_dir_path.mkdir(exist_ok=True)
    feature_path = f'{feature_label_dir_path}/{feature_type}_{feat}_{length}s_features.npy'
    label_path = f'{feature_label_dir_path}/{feature_type}_{feat}_{length}s_labels_.npy'

    if Path(feature_path).exists() and Path(label_path):
        feature_list_master = np.load(feature_path)
        label_list_master = np.load(label_path)

    else:
        print('Loading Features')
        label_list_master = []
        feature_list_master = []

        for file in progress_bar(Path(filepath).rglob('*.wav')):
            for audio_list, label_list in load_audio_generator(file, sample_rate, length, multi_channel):
                for audio, label in zip(audio_list, label_list):
                    label_list_master.append(label)

                    # process data
                    for pro in process_list:
                        if pro == 'normalize':
                            audio = process.normalize(audio)
                        if pro == 'compression':
                            audio = process.compression(audio)
                        if pro == 'noise_reduction':
                            pass

                    features = extract_feature(audio, feature_type, feature_params)
                    feature_list_master.append(features)

        # return features_array, labels, metadata
        feature_list_master = np.array(feature_list_master)
        feature_list_master = np.squeeze(feature_list_master, axis=1)
        feature_list_master = feature_list_master[..., np.newaxis]

        np.save(feature_path, feature_list_master)
        np.save(label_path, label_list_master)

    # return feature_list_master, label_list_master
    return feature_list_master, label_list_master

def load_audio_generator(filepath, sample_rate, length, multi_channel):
    audio = Audio_Abstract(filepath=filepath, sample_rate=sample_rate)
    audio_list_master = []
    label_list_master = []

    multi_channel_options = ['ch_1', 'ch_n', 'split_ch', 'mix_mono', 'original']
    if multi_channel == multi_channel_options[0]:
        pass
    elif multi_channel == multi_channel_options[1]:
        pass
    elif multi_channel == multi_channel_options[2]:
        pass
    elif multi_channel == multi_channel_options[2]:
        pass
    else:
        pass

    if audio.num_channels == 1:
        audio_list, label_list = process.generate_chunks(audio, length=length)
        for audio, label in zip(audio_list, label_list):
            audio_list_master.append(audio)
            label_list_master.append(label)

    else:  # it's 4 channel
        channel_list = process.channel_to_objects(audio)
        for channel in channel_list:
            audio_list, label_list = process.generate_chunks(audio, length=length)
            for audio, label in zip(audio_list, label_list):
                audio_list_master.append(audio)
                label_list_master.append(label)

    yield audio_list_master, label_list_master

def extract_feature(audio, feature_type, feature_params):
    if feature_type == 'spectral':
        return process.spectrogram(audio, feature_params=feature_params)
    elif feature_type == 'mfcc':
        return process.mfcc(audio, feature_params=feature_params)
    elif feature_type == 'filter1':
        return process.custom_filter_1(audio)
    elif feature_type == 'zcr':
        return process.zcr(audio)
    else: raise Exception('Error with feature type')


if __name__ == '__main__':
    timing_stats = time_class(name='Load Features')
    filepath = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/'
                    '1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2')
    length = [10, 8, 6, 4, 2]
    sample_rate = [20_000, 48_000]
    multi_channel = ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
    process_list = ['normalize'] # add labels to list in order to create new processing chain
    feature_type = ['spectral', 'mfcc']
    window_sizes = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256]
    hop_sizes = [1024, 512, 256, 128]
    feature_params = {'bandwidth':(70, 5000), 'window_size':window_sizes[4], 'hop_size':hop_sizes[1]}  # Spectrum
    # feature_params = {'n_coeffs':150}           # MFCC


    features, labels = load_features(filepath, length[2], sample_rate[0], multi_channel[0],
                  process_list, feature_type[0], feature_params)

    total_runtime = timing_stats.stats()

