# File to take isolated sound samples are create a dataset to train a CNN model

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af

from tqdm import tqdm as progress_bar
import numpy as np
from pathlib import Path
import random

def generate_synthetic_data(noise_floor_path, target_path, new_path, sample_length, sample_rate, noise_floor_level, range_of_target_sound):
    Path(new_path).mkdir(exist_ok=True)

    for nf_file in progress_bar(Path(noise_floor_path).iterdir()):
        noise_floor = Audio_Abstract(filepath=nf_file, sample_rate=sample_rate)
        # noise_floor.waveform()

        noise_floor_chunk_list, _ = process.generate_chunks(noise_floor, length=sample_length)

        for file in progress_bar(Path(target_path).iterdir()):
            if 'wav' in file.suffix:
                target = Audio_Abstract(filepath=file, sample_rate=sample_rate)

                if target.sample_length is None:  # or target.sample_length < sample_length
                    continue
                if target.num_channels > 1:
                    channel_list = process.channel_to_objects(target)
                    target_new = process.mix_to_mono(channel_list)
                    target_new.path = target.path
                    target_new.name = target.name
                    target = target_new
                target_chunk_list, _ = process.generate_chunks(target, length=sample_length)

                normalization_values = list(np.arange(range_of_target_sound[0], range_of_target_sound[1], range_of_target_sound[2]))

                for value in normalization_values:
                    # target = process.normalize(target_chunk_list[random.randint(0, len(target_chunk_list) - 1)], percentage=100)
                    target = process.normalize(target_chunk_list[0], percentage=100)
                    target = process.normalize(target, percentage=value)
                    noise_floor_sample = noise_floor_chunk_list[random.randint(0, len(noise_floor_chunk_list) - 1)]
                    noise_floor_edit = process.normalize(noise_floor_sample, percentage=100)
                    noise_floor_edit = process.normalize(noise_floor_edit, percentage=noise_floor_level)
                    mix = process.mix_to_mono([target, noise_floor_edit])
                    mix.name = f'{target.name}_{noise_floor.name}_mix'

                    mix.export(filepath=new_path, name=f'{mix.name}_{value}')

if __name__ == '__main__':

    synthetic_directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Synthetic'
    range_of_target_sound = (5, 71, 1)
    noise_floor_level = 100
    sample_rate = 24_000
    sample_length = 55

    # Diesel Samples
    mix_num = 3
    noise_floor_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Combinations/static 3/backgrounds'
    target_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Combinations/static 3/target'
    new_path = synthetic_directory + f'/static_3_mix_{mix_num}'

    generate_synthetic_data(noise_floor_path, target_path, new_path, sample_length, sample_rate, noise_floor_level, range_of_target_sound)
















