# loading data for CNN models

from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process
from Acoustic.utils import time_class

from tqdm import tqdm as progress_bar
from pathlib import Path
import numpy as np

# Main File for Loading Features
def load_features(filepath, length, sample_rate, multi_channel, chunk_type, process_list, feature_type, feature_params):
    print('Loading Features')

    check_inputs(filepath, length, sample_rate, feature_type, feature_params)

    # returns true is file exists
    if check_if_data_exists(filepath, length, feature_type, feature_params):
        print('Features Exist')
        feature_path, label_path, _ = feature_labels_file_names(filepath, length, feature_type, feature_params)

    else:
        print('Creating Features')
        label_list_master = []
        feature_list_master = []
        audio_name_master = []

        for file in progress_bar(Path(filepath).rglob('*.wav')):
            for audio_list, label_list in load_audio_generator(file, sample_rate, length, multi_channel, chunk_type):
                for audio, label in zip(audio_list, label_list):
                    label_list_master.append(label)
                    audio = preprocess_files(audio, process_list)
                    features = extract_feature(audio, feature_type, feature_params)
                    feature_list_master.append(features)
                    audio_name_master.append(file.stem)

        # return features_array, labels, metadata
        feature_list_master = format_features(feature_list_master)
        feature_stats = stats_file_create(feature_list_master, length, feature_type, feature_params, sample_rate, multi_channel, filepath, process_list)

        feature_path, label_path, audio_names_path = feature_labels_file_names(filepath, length, feature_type, feature_params)
        feature_stat_path = (audio_names_path.split('.')[0].split('_')[:-1])
        feature_stat_path = f"{'_'.join(feature_stat_path)}_{Path(filepath).stem}_stats.txt"

        np.save(feature_path, feature_list_master)
        np.save(label_path, label_list_master)
        write_filenames_to_file(audio_name_master, audio_names_path)
        write_filenames_to_file(feature_stats, feature_stat_path)

    feature_list_master = np.load(feature_path, mmap_mode='r')
    label_list_master = np.load(label_path, mmap_mode='r')
    label_list_master = np.array(label_list_master)
    label_list_master = np.squeeze(label_list_master)

    return feature_list_master, label_list_master

# Function to generator audio files and keep memory usage low
def load_audio_generator(filepath, sample_rate, length, multi_channel, chunking_type):

    audio_list_master = []
    label_list_master = []

    multi_channel_options = ['ch_1', 'ch_n', 'split_ch', 'mix_mono', 'original', '4ch']
    if multi_channel.lower() == multi_channel_options[0]:
        audio = Audio_Abstract(filepath=filepath, sample_rate=sample_rate, num_channels=1)
    elif multi_channel.lower() == multi_channel_options[1]:
        pass
    elif multi_channel.lower() == multi_channel_options[2]:
        pass
    elif multi_channel.lower() == multi_channel_options[3]:
        audio = Audio_Abstract(filepath=filepath, sample_rate=sample_rate)
        channel_list = process.channel_to_objects(audio)
        audio = process.mix_to_mono(audio for audio in channel_list)
    elif multi_channel.lower() == multi_channel_options[4]:
        audio = Audio_Abstract(filepath=filepath, sample_rate=sample_rate)
    else:
        audio = Audio_Abstract(filepath=filepath, sample_rate=sample_rate, num_channels=4)

    if chunking_type == 'window':
        if audio.num_channels == 1:
            audio_list, label_list = process.generate_windowed_chunks(audio, window_size=length)
            for audio, label in zip(audio_list, label_list):
                audio_list_master.append(audio)
                label_list_master.append(label)

        else:  # it's 4 channel
            # group features as chunk1[a, b, c, d], chunk2[a, b, c, d], etc

            channel_list = process.channel_to_objects(audio)
            for channel in channel_list:
                audio_list, label_list = process.generate_windowed_chunks(channel, window_size=length)
                for audio, label in zip(audio_list, label_list):
                    audio_list_master.append(audio)
                    label_list_master.append(label)
                    # print(f'Channel: {audio.which_channel}\t|\tChunk Index: {audio.chunk_index}\t|\tChunk Time: {audio.chunk_time}')

    else:
        if audio.num_channels == 1:
            audio_list, label_list = process.generate_chunks(audio, length=length)
            for audio, label in zip(audio_list, label_list):
                audio_list_master.append(audio)
                label_list_master.append(label)

        else:  # it's 4 channel
            channel_list = process.channel_to_objects(audio)
            for channel in channel_list:
                audio_list, label_list = process.generate_chunks(channel, length=length)
                for audio, label in zip(audio_list, label_list):
                    audio_list_master.append(audio)
                    label_list_master.append(label)

    yield audio_list_master, label_list_master

# Function to preprocess audio files
def preprocess_files(audio_object, processes):
    audio = audio_object
    for pro in processes:
        if pro == 'normalize':
            audio = process.normalize(audio)
        if pro == 'compression':
            audio = process.compression(audio)
        if pro == 'noise_reduction':
            pass

    return audio

# Function for Extracting Features from audio object
def extract_feature(audio, feature_type, feature_params):
    if feature_type == 'spectral':
        return process.spectrogram_2(audio, feature_params=feature_params)
    elif feature_type == 'mfcc':
        return process.mfcc(audio, feature_params=feature_params)
    elif feature_type == 'feature_combo_1':
        return process.feature_combo_1(audio)
    elif feature_type == 'zcr':
        return process.zcr(audio)
    else: raise Exception('Error with feature type')

# Function to format features for CNN model
def format_features(feature_list):
    feature_list_format = np.array(feature_list)
    feature_list_format = np.squeeze(feature_list_format)
    feature_list_format = feature_list_format[..., np.newaxis]
    return feature_list_format

# Function for Input Checking
def check_inputs(filepath, length, sample_rate, feature_type, feature_params):

    if not Path(filepath).exists():
        print(filepath)
        raise Exception('Directory does not exists')
    if type(length) is not int:
        raise Exception('Length needs to be an integer')
    if length < 1:
        raise Exception('Length needs to be greater than 1')
    if type(sample_rate) is not int:
        raise Exception('Sample Rate needs to be an integer')
    if sample_rate < 10_000 or sample_rate > 48_0001:
        raise Exception('Sample Rate Out of Range')
    if feature_params != 'None':
        if feature_type == 'spectral':
            if type(feature_params.get('bandwidth')[0]) is not int or type(feature_params.get('bandwidth')[1]) is not int:
                raise Exception('Bandwidth must be integers')
            # if feature_params.get('bandwidth')[0] < 50 or feature_params.get('bandwidth')[1] > int(sample_rate/2):
            #     raise Exception('Bandwidth is out of range')
            # if feature_params.get('window_size') % 2 != 0:
            #     raise Exception('Window Size needs to be a power of 2')
        if feature_type == 'mfcc':
            if type(feature_params.get('n_coeffs')) is not int:
                raise Exception('Number of Coefficients must be integer')

# Function to create the features / labels / audio names file names for storage
def feature_labels_file_names(filepath, length, feature_type, feature_params):
    # Parsing Feature Parameters for Saving Data after Processing
    feat = 'None'
    if feature_params != 'None':
        if feature_type == 'spectral':
            bandwidth = feature_params.get('bandwidth')
            nperseg = feature_params.get('nperseg')
            feat = f'{bandwidth[0]}-{bandwidth[1]}-{nperseg}'
        if feature_type == 'mfcc':
            feat = feature_params.get('n_coeffs')
        if feature_type == 'feature_combo_1':
            feat = 'None'


    # Make features_label folder if doesnt exist
    feature_label_dir_path = Path(f'{Path.cwd()}/features_labels')
    feature_label_dir_path.mkdir(exist_ok=True)

    dataset = Path(filepath)

    feature_path = f'{feature_label_dir_path}/{feature_type}_{feat}_{length}s_{dataset.stem}_features.npy'
    label_path = f'{feature_label_dir_path}/{feature_type}_{feat}_{length}s_{dataset.stem}_labels_.npy'
    audio_names_path = f'{feature_label_dir_path}/{feature_type}_{feat}_{length}s_{dataset.stem}_features_files.txt'

    return feature_path, label_path, audio_names_path

# Function to see if data request already exists or needs to be created
def check_if_data_exists(filepath, length, feature_type, feature_params):

    feature_path, label_path, audio_names_path = feature_labels_file_names(filepath, length, feature_type, feature_params)

    # If files wanted already exist, return them instead of loading data again
    if Path(feature_path).exists() and Path(label_path).exists():
        audio_name_master = []
        for file in Path(filepath).rglob('*.wav'):
            audio_name_master.append(file.stem)
        if not Path(audio_names_path).exists():
            write_filenames_to_file(audio_name_master, audio_names_path)

        with open(audio_names_path, 'r') as f:
            filenames_from_file = set(line.strip() for line in f)

        if set(audio_name_master) == filenames_from_file:
            return True
        else: return False
    else: return False

# Function to write a list to a text file
def write_filenames_to_file(filenames, output_file):
    """
    Write each filename from a list to a new line in an output file.

    :param filenames: List of filenames.
    :param output_file: The name of the output file.
    """

    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')

# Functiion to get stats and write them to a file
def stats_file_create(feature_list, length, feature_type, feature_params, sample_rate, multi_channel, filepath, process_list):

    feat = 'None'
    if feature_type == 'spectral':
        bandwidth = feature_params.get('bandwidth')
        nperseg = feature_params.get('nperseg')
        feat = f'Bandwidth: ({bandwidth[0]}-{bandwidth[1]}) / nperseg: {nperseg}'
    if feature_type == 'mfcc':
        feat = feature_params.get('n_coeffs')
        feat = f'Num Coeffs: {feat}'

    feat_type = f'Feature Type: {feature_type.upper()}'
    params = f'Feature Parameters: {feat}'
    sr = f'Sample Rate: {sample_rate} Hz'
    len = f'Sample Length: {length} sec'
    shape = f'Shape: {feature_list.shape}'
    max = f'Max: {feature_list.max()}'
    min = f'Min: {feature_list.min()}'
    mean = f'Mean: {feature_list.mean()}'
    std = f'Std: {feature_list.std()}'
    multch = f'Multi Channel: {multi_channel.title()}'
    path = f'Filepath: {filepath}'
    pro_list = f'Processes Applied: {process_list}'

    stats_list = [path, multch, sr, len, pro_list, feat_type, params, shape, max, min, mean, std]

    return stats_list


if __name__ == '__main__':
    timing_stats = time_class(name='Load Features')
    filepath = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/'
                    '1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2')
    length = [2, 4, 6, 8, 10]
    sample_rate = [12_000, 18_000, 24_000, 36_000, 48_000]
    multi_channel = ['original', 'ch_1', 'ch_n', 'split_ch', 'mix_mono']
    chunk_type = ['regular', 'window']
    process_list = ['normalize']  # add labels to list in order to create new processing chain
    feature_type = ['spectral', 'mfcc']
    window_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    hop_sizes = [128, 256, 512, 1024]
    feature_params = {'bandwidth':(70, 20000), 'nperseg':window_sizes[0], 'hop_size':hop_sizes[2]}  # Spectrum
    # feature_params = {'n_coeffs': 100}  # MFCC


    features, labels = load_features(filepath,
                                     length[4],
                                     sample_rate[4],
                                     multi_channel[0],
                                     chunk_type[0],
                                     process_list,
                                     feature_type[0],
                                     feature_params)



    total_runtime = timing_stats.stats()

