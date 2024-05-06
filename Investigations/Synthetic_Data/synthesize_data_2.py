# File to take isolated sound samples are create a dataset to train a CNN model

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af

from tqdm import tqdm as progress_bar
import numpy as np
from pathlib import Path
import random


def calculate_power(audio):
    """Calculate the power of an audio signal."""
    return np.sum(audio ** 2) / len(audio)

def find_norm_value(noise, target, value):
    noise_norm_value = value
    target_norm_value = value
    precision_value = 5
    epsilon = 1e-5
    normalized_noise = process.normalize(noise, percentage=noise_norm_value)

    while True:

        normalized_target = process.normalize(target, percentage=target_norm_value)

        noise_power = np.round(calculate_power(normalized_noise.data), precision_value)
        target_power = np.round(calculate_power(normalized_target.data), precision_value)
        # print(f'Looking... {noise_power} | {target_power}')
        # noise_power = np.round(calculate_RMS(normalized_noise.data), precision_value)
        # target_power = np.round(calculate_RMS(normalized_target.data), precision_value)

        if abs(noise_power - target_power) < epsilon:
            break
        else:
            if noise_power > target_power:
                target_norm_value += 0.01
            else:
                target_norm_value -= 0.01

    return target_norm_value

def calculate_distance_for_spl(spl_initial, distance_initial, spl_desired):
    distance_desired = distance_initial * 10 ** ((spl_initial - spl_desired) / 20)
    return distance_desired

def amplitude_for_distance(original_spl, d1, d2):
    """
    Adjust the amplitude of a signal based on the change in distance from the sound source.

    Parameters:
    - signal: The original digital audio signal.
    - original_spl: The SPL at the original distance d1.
    - d1: The original distance from the sound source.
    - d2: The new distance from the sound source.

    Returns:
    - The adjusted digital audio signal.
    """
    # Calculate the change in SPL
    delta_spl = 20 * np.log10(d1 / d2)

    # Adjust the SPL for the new distance
    new_spl = original_spl + delta_spl

    # Assuming a direct relationship, calculate the scaling factor for amplitude
    # This is a simplified model and may need adjustment for accurate real-world applications
    scale_factor = 10 ** ((new_spl - original_spl) / 20)
    # print(f'Scale Factor: {scale_factor}')

    return scale_factor

def generate_synthetic_data(noise_floor_path, target_path, new_path, sample_length, sample_rate, noise_floor_SPL, target_SPL, target_distance, range_of_target_sound):
    Path(new_path).mkdir(exist_ok=True)

    noise_floor = Audio_Abstract(filepath=noise_floor_path, sample_rate=sample_rate)
    # noise_floor.waveform()

    noise_floor_chunk_list, _ = process.generate_chunks(noise_floor, length=sample_length)

    for file in progress_bar(Path(target_path).iterdir()):
        if 'wav' in file.suffix:
            target = Audio_Abstract(filepath=file, sample_rate=sample_rate)

            # Preprocessing of Target File
            if target.sample_length is None or target.sample_length < sample_length:
                continue
            if target.num_channels > 1:
                channel_list = process.channel_to_objects(target)
                target_new = process.mix_to_mono(channel_list)
                # target_new = process.normalize(target_new, 98)
                target_new.path = target.path
                target_new.name = target.name
                target = target_new

            target_chunk_list, _ = process.generate_chunks(target, length=sample_length)

            noise_norm_val = 90
            normalized_noise = process.normalize(noise_floor_chunk_list[0], percentage=noise_norm_val)
            target_norm_val = find_norm_value(noise_floor_chunk_list[0], target_chunk_list[0], noise_norm_val)

            # Set all norm values for each
            
            normalized_target = process.normalize(target, percentage=target_norm_val)

            # Generate the scale factors for particular file
            new_target_distance = calculate_distance_for_spl(target_SPL, target_distance, noise_floor_SPL)
            distances = list(np.arange(range_of_target_sound[0], range_of_target_sound[1], range_of_target_sound[2]))
            scale_factors = [amplitude_for_distance(noise_floor_SPL, new_target_distance, distance) for distance in distances]


            for value in scale_factors:
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

    # Directory where new synthetic dataset directory will be created
    synthetic_directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Synthetic'

    # Audio Specs
    sample_rate = 24_000
    sample_length = 10

    # Samples used to synthesize dataset
    noise_floor_path = af.hex_hover_combo_thick
    target_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Combinations/static 3'

    # Model Specs
    noise_SPL = 98
    target_SPL = 93 #dB(C)
    target_distance = 12 #m
    range_of_target_sound = (10, 60, 0.5)  # in meters (start distance, end distance, interval

    # noise_floor_level = 100


    # Diesel Samples
    mix_num = 7

    new_path = synthetic_directory + f'/diesel_hex_mix_{mix_num}'


    generate_synthetic_data(noise_floor_path, target_path, new_path, sample_length, sample_rate, noise_floor_SPL, target_SPL, target_distance, range_of_target_sound)
















