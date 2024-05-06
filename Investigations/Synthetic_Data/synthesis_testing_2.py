

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass


def calculate_power(audio):
    """Calculate the power of an audio signal."""
    return np.sum(audio ** 2) / len(audio)

def calculate_RMS(audio):
    """Calculate the power of an audio signal."""
    return np.sqrt(np.sum(audio ** 2) / len(audio))
    # return np.sum(audio ** 2) / len(audio)

def find_norm_value(value):
    ratio_found = False
    noise_norm_value = value
    target_norm_value = value
    precision_value = 6
    epsilon = 1e-6
    normalized_noise = process.normalize(noise, percentage=noise_norm_value)

    while not ratio_found:
        normalized_target = process.normalize(target, percentage=target_norm_value)

        noise_power = np.round(calculate_power(normalized_noise.data), precision_value)
        target_power = np.round(calculate_power(normalized_target.data), precision_value)

        # noise_power = np.round(calculate_RMS(normalized_noise.data), precision_value)
        # target_power = np.round(calculate_RMS(normalized_target.data), precision_value)

        if abs(noise_power - target_power) < epsilon:
            ratio_found = True
        else:
            if noise_power > target_power:
                target_norm_value += 0.1
            else:
                noise_norm_value += 0.1

    norm_difference = target_norm_value - noise_norm_value

    return norm_difference

def find_noise_target_norm_relationship():
    norm_list = [x for x in range(1, 101)]
    values_list = [find_norm_value(x) for x in norm_list]

    # Perform linear regression: deg=1 for linear
    slope, intercept = np.polyfit(norm_list, values_list, deg=1)

    # Create a linear model function based on the slope and intercept
    linear_model = np.poly1d([slope, intercept])

    # Generate y values based on the model to plot or analyze
    # y_model = linear_model(norm_list)
    # y_model represents how much greater the target norm value is to noise norm value on full norm scale

    return linear_model

def get_target_value_based_on_noise_value_for_one_to_one(noise_val):
    # this is for 1:1 SNR
    linear_function = find_noise_target_norm_relationship()
    addition = linear_function(noise_val)
    tar_val = noise_val + addition
    return np.round(tar_val, 3)

def square_law_equation(measured_SPL, distance_of_measured_SPL, input_distance):
    # measured_SPL is measured in dB(C)
    # distance_of_measured_SPL and input_distance in meters

    desired_SPL = np.round((measured_SPL - 20 * (np.log10(input_distance / distance_of_measured_SPL))), 5)

    return desired_SPL



if __name__ == '__main__':
    noise_floor_path = af.hex_hover_combo_thick
    # noise_floor_path = af.angel_wind_b
    # noise_floor_path = af.amb_campus_1
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

    # -----------------------------------------------------
    # noise_power = calculate_power(noise.data)
    # target_power = calculate_power(target.data)
    #
    # print(f'Noise Power: {noise_power}')
    # print(f'Target Power: {target_power}')
    #
    # noise_RMS = calculate_RMS(noise.data)
    # target_RMS = calculate_RMS(target.data)
    #
    # print(f'Noise RMS: {noise_RMS}')
    # print(f'Target RMS: {target_RMS}')
    #
    # bars = ['npwr', 'tpwr', 'nrms', 'trms']
    # bar_list = [noise_power, target_power, noise_RMS, target_RMS]
    #
    # plt.bar(bars, bar_list)
    # plt.show()

    # -----------------------------------------------------
    # normalized_noise = process.normalize(noise)
    # normalized_target = process.normalize(target)
    #
    # noise_power = calculate_power(normalized_noise.data)
    # target_power = calculate_power(normalized_target.data)
    #
    # print(f'Noise Power: {noise_power}')
    # print(f'Target Power: {target_power}')
    #
    # noise_RMS = calculate_RMS(normalized_noise.data)
    # target_RMS = calculate_RMS(normalized_target.data)
    #
    # print(f'Noise RMS: {noise_RMS}')
    # print(f'Target RMS: {target_RMS}')
    #
    # bars = ['npwr', 'tpwr', 'nrms', 'trms']
    # bar_list = [noise_power, target_power, noise_RMS, target_RMS]
    #
    # plt.bar(bars, bar_list)
    # plt.show()

    # -----------------------------------------------------
    # Same Length must be same
    # Calculate Power
    # Compare
    # Find Norm values that make power equal
    # Adjust

    # ratio_found = False
    # noise_norm_value = 80
    # target_norm_value = 80
    # precision_value = 4
    #
    # while not ratio_found:
    #     normalized_noise = process.normalize(noise, percentage=noise_norm_value)
    #     normalized_target = process.normalize(target, percentage=target_norm_value)
    #
    #     # noise_power = np.round(calculate_power(normalized_noise.data), precision_value)
    #     # target_power = np.round(calculate_power(normalized_target.data), precision_value)
    #
    #     noise_power = np.round(calculate_RMS(normalized_noise.data), precision_value)
    #     target_power = np.round(calculate_RMS(normalized_target.data), precision_value)
    #
    #     if noise_power == target_power:
    #         ratio_found = True
    #     else:
    #         if noise_power > target_power:
    #             target_norm_value += 0.1
    #         else:
    #             noise_norm_value -= 0.1
    #
    #
    # print(f'Noise Value: {noise_norm_value}')
    # print(f'Target Value: {target_norm_value}')
    #
    # norm_difference = target_norm_value - noise_norm_value
    # print(f'Diff: {norm_difference}')
    #
    # norm_list = [x for x in range(1, 101)]
    # values_list = [find_norm_value(x) for x in norm_list]
    #
    # # Perform linear regression: deg=1 for linear
    # slope, intercept = np.polyfit(norm_list, values_list, deg=1)
    #
    # # Create a linear model function based on the slope and intercept
    # linear_model = np.poly1d([slope, intercept])
    #
    # # Generate y values based on the model to plot or analyze
    # y_model = linear_model(norm_list)
    #
    # # y_model represents how much greater the target norm value is to noise norm value on full norm scale
    #
    # plt.plot(norm_list, values_list)
    # plt.plot(norm_list, y_model)
    # plt.show()

    # -----------------------------------------------------
    # Find norm value for 1:1 SNR
    # noise_norm_val = 50
    # target_norm_val = get_target_value_based_on_noise_value_for_one_to_one(noise_norm_val)
    # print(f'Noise Norm Val: {noise_norm_val} / Tar Norm Val: {target_norm_val}')
    #
    # normalized_noise = process.normalize(noise, percentage=noise_norm_val)
    # normalized_target = process.normalize(target, percentage=target_norm_val)
    #
    # normalized_noise.waveform(display=True)
    # normalized_target.waveform(display=True)

    # -----------------------------------------------------
    # Now that found 1:1 SNR, how can this be adjusted to mimic an increasing distance from target?
    # The power of the noise can be known: if ambient, then SPL meter, if UAV, then experiments showed theirs
    # Angel: 105dB
    # Hex: 98 dB

    # if SPL value for ego are known and norm value assigned, then find ref value that relates SPL to RMS
    noise_norm_val = 50
    hex_SPL_value = 98
    normalized_noise = process.normalize(noise, percentage=noise_norm_val)
    hex_rms_value_at_known_norm_value = calculate_RMS(normalized_noise.data)
    p_ref = hex_rms_value_at_known_norm_value / np.power(10, hex_SPL_value / 20)

    # Now that I know the reference pressure level, this can be applied to the target sample
    # to relate the SPL level to a norm value

    # target norm val is the norm val where targets RMS is same as noise which is 98dB
    target_norm_val = get_target_value_based_on_noise_value_for_one_to_one(noise_norm_val)

    # SPL values at each distance for target
    distances = [x for x in range(10, 101, 1)]
    target_distance_of_measured_SPL = 12  # m
    target_measured_SPL = 93  # dB(C)
    target_SPL_values = [square_law_equation(
        target_measured_SPL,
        target_distance_of_measured_SPL,
        distance) for distance in distances]

    # Using the reference value found, calculate RMS pressures for each SPL value
    rms_pressure_list = [(p_ref * np.power(10, SPL_value / 20)) for SPL_value in target_SPL_values]
    plt.plot(distances, target_SPL_values)
    plt.show()
    plt.plot(distances, rms_pressure_list)
    plt.show()

    # Find norm value that makes the sample at each pressure level which relates the norm value to a distance






    # Maybe find norm equation that mimics square law distance equation?
    # Final function would have range of distances from target as input and give dataset based on that

    # would need estimates for noise power and target power

    # This method is for if target intensity is less than noise floor

    # distances = [x for x in range(2, 101, 1)]
    # target_distance_of_measured_SPL = 12  # m
    #
    # target_measured_SPL = 93  # dB(C)
    # target_SPL_values_L = [square_law_equation(
    #     target_measured_SPL,
    #     target_distance_of_measured_SPL,
    #     distance) for distance in distances]
    #
    # target_measured_SPL = 90  # dB(C)
    # target_SPL_values_m = [square_law_equation(
    #     target_measured_SPL,
    #     target_distance_of_measured_SPL,
    #     distance) for distance in distances]
    #
    # target_measured_SPL = 87  # dB(C)
    # target_SPL_values_s = [square_law_equation(
    #     target_measured_SPL,
    #     target_distance_of_measured_SPL,
    #     distance) for distance in distances]
    #
    # # print(target_SPL_values)
    # plt.figure(figsize = (10, 6))
    # plt.plot(distances, target_SPL_values_L, label='Target Loud')
    # plt.plot(distances, target_SPL_values_m, label='Target Medium')
    # plt.plot(distances, target_SPL_values_s, label='Target Soft')
    # plt.axhline(y=105, color='g', linestyle='--', label='Angel Ego')
    # plt.axhline(y=98, color='b', linestyle='--', label='Hex Ego')
    # plt.axhline(y=64, color='black', linestyle='--', label='ambient noise floor')
    # plt.axvline(x=30, color='r', linestyle='--', label='target range')
    # plt.axvline(x=50, color='r', linestyle='--', label='target range')
    # plt.legend(loc='best')
    # plt.show()

    # target profile = (SPL , distance_measurement) (dB(C) , meters)

    # Point where target and UAV power is equal would be distance where SPL's are the same

    # What distance are they equal?

    # distances = [x for x in np.arange(2, 101, 0.1)]
    # target_distance_of_measured_SPL = 12  # m
    # target_measured_SPL = 93  # dB(C)
    # target_SPL_values_L = [square_law_equation(
    #     target_measured_SPL,
    #     target_distance_of_measured_SPL,
    #     distance) for distance in distances]
    #
    # noise_floor_SPL = 98
    # distance_equals_index = min(enumerate(target_SPL_values_L), key=lambda x: abs(x[1] - noise_floor_SPL))[0]
    # # print(distance_equals_index)
    #
    # distance_where_equal = np.round(distances[distance_equals_index], 3)
    # print(f'Distance where Equal: {distance_where_equal} meters')
    #
    # # Find norm value for 1:1 SNR
    # noise_norm_val = 90 # arbitrary value
    # target_norm_val = get_target_value_based_on_noise_value_for_one_to_one(noise_norm_val)
    # print(f'Noise Norm Val: {noise_norm_val} / Tar Norm Val: {target_norm_val}')
    #
    # normalized_noise = process.normalize(noise, percentage=noise_norm_val)
    # normalized_target = process.normalize(target, percentage=target_norm_val)
    #
    # normalized_noise.waveform(display=True)
    # normalized_target.waveform(display=True)









