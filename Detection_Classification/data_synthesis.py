# File to take isolated sound samples are create a dataset to train a CNN model

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process

import numpy as np

uav_options = ['hex', 'angel', 'penguin']
noise_options = ['ambient', 'wind']
target_options = ['diesel', 'gas']

sample_length = 10

# What if I want to do this for multiple ambient environments?
noise_floor_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Random/home 2.wav'
noise_floor = Audio_Abstract(filepath=noise_floor_path)

# noise_floor = process.compression(noise_floor)
noise_floor = process.normalize(noise_floor, percentage=95)
noise_floor_chunk_list = process.generate_chunks(noise_floor, length=sample_length, training=False)

target_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/Campus/Construction 2/construction 2-1.wav'
target = Audio_Abstract(filepath=target_path)
if target.sample_length < 10:
    print('Go to Next Target Sample')
target_chunk_list = process.generate_chunks(target, length=sample_length, training=False)
target_index = int(len(target_chunk_list)/2)
target = target_chunk_list[target_index]

# target = process.compression(target)
normalization_values = list(np.arange(95, 45, -5))

export_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Engine vs Ambience/dataset 2/1'
for value in normalization_values:
    target = process.normalize(target, percentage=value)
    noise = noise_floor_chunk_list[0]
    mix = process.to_mono(target, noise)
    # mix.export(filepath = export_path, name = f'{target.name}_{value}')























if __name__ == '__main__':
    isolated_sample_directory = ''

