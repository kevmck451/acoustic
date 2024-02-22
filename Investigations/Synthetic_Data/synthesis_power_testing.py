

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af


import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import random
import time
import matplotlib.pyplot as plt


def calculate_power(audio):
    """Calculate the power of an audio signal."""
    return np.sum(audio ** 2) / len(audio)

def find_norm_value(value):
    ratio_found = False
    noise_norm_value = value
    target_norm_value = value
    precision_value = 5

    while not ratio_found:
        normalized_noise = process.normalize(noise, percentage=noise_norm_value)
        normalized_target = process.normalize(target, percentage=target_norm_value)

        noise_power = np.round(calculate_power(normalized_noise.data), precision_value)
        target_power = np.round(calculate_power(normalized_target.data), precision_value)

        if noise_power == target_power:
            ratio_found = True
        else:
            if noise_power > target_power:
                target_norm_value += 0.1
            else:
                noise_norm_value += 0.1


    norm_difference = target_norm_value - noise_norm_value

    return norm_difference


if __name__ == '__main__':
    noise_floor_path = af.hex_hover_combo_thick
    sample_rate = 24_000
    sample_length = 10

    noise_sample = Audio_Abstract(filepath=noise_floor_path, sample_rate=sample_rate)
    # print(noise_sample.stats())
    # print(noise_sample.data.dtype)
    # print(noise_sample)
    # noise_floor.waveform(display=True)

    target_path = af.diesel_tank_1_3
    target_sample = Audio_Abstract(filepath=target_path, sample_rate=sample_rate)
    # print(target_sample.stats())
    # print(target_sample.data.dtype)
    # print(target_sample)
    # target.waveform(display=True)

    noise_chunk_list, _ = process.generate_chunks(noise_sample, length=sample_length)
    target_chunk_list, _ = process.generate_chunks(target_sample, length=sample_length)

    noise = noise_chunk_list[0]
    target = target_chunk_list[0]
    # print(noise)
    # print(target)

    # noise_power = calculate_power(noise.data)
    # target_power = calculate_power(target.data)
    #
    # print(f'Noise Power: {noise_power}')
    # print(f'Target Power: {target_power}')
    #
    # normalized_noise = process.normalize(noise)
    # normalized_target = process.normalize(target)
    #
    # noise_power = calculate_power(normalized_noise.data)
    # target_power = calculate_power(normalized_target.data)
    #
    # print(f'Noise Norm Power: {noise_power}')
    # print(f'Target Norm Power: {target_power}')

    # Same Length must be same
    # Calculate Power
    # Compare
    # Find Norm values that make power equal
    # Adjust

    # ratio_found = False
    # noise_norm_value = 20
    # target_norm_value = 20
    # precision_value = 4
    #
    # while not ratio_found:
    #     normalized_noise = process.normalize(noise, percentage=noise_norm_value)
    #     normalized_target = process.normalize(target, percentage=target_norm_value)
    #
    #     noise_power = np.round(calculate_power(normalized_noise.data), precision_value)
    #     target_power = np.round(calculate_power(normalized_target.data), precision_value)
    #
    #     if noise_power == target_power:
    #         ratio_found = True
    #     else:
    #         if noise_power > target_power:
    #             target_norm_value += 0.1
    #         else:
    #             noise_norm_value += 0.1
    #
    #
    # print(f'Noise Value: {noise_norm_value}')
    # print(f'Target Value: {np.round(target_norm_value, 1)}')
    #
    # norm_difference = np.round(noise_norm_value - np.round(target_norm_value, 1), 1)
    # print(f'Diff: {norm_difference}')



    norm_list = [x for x in range(1, 101)]
    values_list = [find_norm_value(x) for x in norm_list]

    # Perform linear regression: deg=1 for linear
    slope, intercept = np.polyfit(norm_list, values_list, deg=1)

    # Create a linear model function based on the slope and intercept
    linear_model = np.poly1d([slope, intercept])

    # Generate y values based on the model to plot or analyze
    y_model = linear_model(norm_list)

    plt.plot(norm_list, values_list)
    plt.plot(norm_list, y_model)
    plt.show()





