# File to take isolated sound samples are create a dataset to train a CNN model

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process

import numpy as np
from pathlib import Path
import random

synthetic_directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Synthetic'

uav_options = ['hex', 'angel', 'penguin']
noise_options = ['ambient', 'wind']
target_options = ['engine', 'diesel', 'gas']

sample_length = 8

# What if I want to do this for multiple ambient environments?
noise_floor_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Isolated Samples/Ambient/ag_amb_2-1.wav'
noise_floor = Audio_Abstract(filepath=noise_floor_path)
# noise_floor.waveform()

noise_floor = process.normalize(noise_floor, percentage=100)
noise_floor = process.normalize(noise_floor, percentage=50)
noise_floor_chunk_list, _ = process.generate_chunks(noise_floor, length=sample_length)
#
target_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Isolated Samples/Diesel'
# target_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Isolated Samples/Gas'

for file in Path(target_path).iterdir():
    if 'wav' in file.suffix:
        target = Audio_Abstract(filepath=file, sample_rate=20000)

        if target.sample_length is None or target.sample_length < sample_length:
            print(file)
            print('Go to Next Target Sample')
            continue
        if target.num_channels > 1:
            channel_list = process.channel_to_objects(target)
            target = channel_list[0]
        target_chunk_list, _ = process.generate_chunks(target, length=sample_length)

        normalization_values = list(np.arange(10, 60, 2))

        for value in normalization_values:
            target = process.normalize(target_chunk_list[random.randint(0, len(target_chunk_list) - 1)], percentage=100)
            target = process.normalize(target, percentage=value)

            mix = process.mix_to_mono(target, noise_floor_chunk_list[random.randint(0, len(noise_floor_chunk_list)-1)])
            mix.name = f'{target.name}_{noise_floor.name}_mix'

            mix.export(filepath = synthetic_directory, name = f'{mix.name}_{value}')










# List of ambient background already used

# noise_floor_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Isolated Samples/Ambient/home_amb_1_a.wav'
















