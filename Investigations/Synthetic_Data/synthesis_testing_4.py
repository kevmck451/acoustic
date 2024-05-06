

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt



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



if __name__ == '__main__':

    sample_rate = 24_000
    sample_length = 10

    # noise_floor_path = af.hex_hover_combo_thick
    noise_floor_path = af.angel_wind_b
    # noise_floor_path = af.amb_campus_1
    noise_sample = Audio_Abstract(filepath=noise_floor_path, sample_rate=sample_rate)

    target_path = af.diesel_tank_1_3
    target_sample = Audio_Abstract(filepath=target_path, sample_rate=sample_rate)

    noise_chunk_list, _ = process.generate_chunks(noise_sample, length=sample_length)
    target_chunk_list, _ = process.generate_chunks(target_sample, length=sample_length)

    noise = noise_chunk_list[0]
    target = target_chunk_list[0]

    # Find norm value for 1:1 SNR
    # print('Finding Norm Value for 1:1 SNR between UAV and Target')
    noise_norm_val = 50
    target_norm_val = find_norm_value(noise, target, noise_norm_val)
    # print(f'Noise Norm Val: {noise_norm_val} / Tar Norm Val: {target_norm_val}')

    # print('Normalizing Samples so that power is equal')
    normalized_noise = process.normalize(noise, percentage=noise_norm_val)
    normalized_target = process.normalize(target, percentage=target_norm_val)

    hex_SPL_value = 98
    target_measured_SPL = 93  # dB(C)
    target_distance_of_measured_SPL = 12  # m

    # print('Finding Distance where target SPL matches noise SPL')
    new_target_distance = calculate_distance_for_spl(target_measured_SPL, target_distance_of_measured_SPL, hex_SPL_value)

    distances = [x for x in range(2, 81, 1)]

    print('Finding Scalar to Adjust Target to Particular Distance')

    scale_factors = [amplitude_for_distance(hex_SPL_value, new_target_distance, distance) for distance in distances]

    plt.plot(distances, scale_factors)
    plt.show()