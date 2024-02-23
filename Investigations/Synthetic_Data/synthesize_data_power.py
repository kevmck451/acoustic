



from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af


import numpy as np

def calculate_power(audio):
    """Calculate the power of an audio signal."""
    return np.sum(audio ** 2) / len(audio)

def find_norm_value(value):
    ratio_found = False
    noise_norm_value = value
    target_norm_value = value
    precision_value = 6

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

def norm_value_from_distance(target_measured_SPL, target_distance_of_measured_SPL, distance, noise_floor_SPL):
    distance_SPL = square_law_equation(
        target_measured_SPL,
        target_distance_of_measured_SPL,
        distance)

    percentage_of_SPL = (distance_SPL / noise_floor_SPL)
    norm_value = target_norm_val * percentage_of_SPL
    return norm_value

if __name__ == '__main__':

    noise_floor_path = af.hex_hover_combo_thick
    sample_rate = 24_000
    sample_length = 10
    noise_sample = Audio_Abstract(filepath=noise_floor_path, sample_rate=sample_rate)
    target_path = af.diesel_tank_1_3
    target_sample = Audio_Abstract(filepath=target_path, sample_rate=sample_rate)
    noise_chunk_list, _ = process.generate_chunks(noise_sample, length=sample_length)
    target_chunk_list, _ = process.generate_chunks(target_sample, length=sample_length)

    noise = noise_chunk_list[0]
    target = target_chunk_list[0]

    # Find Distance Noise Floor and Target are Equal
    distances = [x for x in np.arange(2, 101, 0.1)]
    target_distance_of_measured_SPL = 12  # m
    target_measured_SPL = 93  # dB(C)
    target_SPL_values_L = [square_law_equation(
        target_measured_SPL,
        target_distance_of_measured_SPL,
        distance) for distance in distances]

    noise_floor_SPL = 98
    distance_equals_index = min(enumerate(target_SPL_values_L), key=lambda x: abs(x[1] - noise_floor_SPL))[0]
    distance_where_equal = np.round(distances[distance_equals_index], 3)
    print(f'Distance where Equal: {distance_where_equal} meters')



    # Relate to Normalize Values
    noise_norm_val = 90  # arbitrary value
    target_norm_val = get_target_value_based_on_noise_value_for_one_to_one(noise_norm_val)
    print(f'Noise Norm Val: {noise_norm_val} / Tar Norm Val: {target_norm_val}')

    # Linearize the dB(C) of target in relation to distance
    distances = [x for x in range(30, 50, 1)]
    norm_values_in_range = [norm_value_from_distance(
        target_measured_SPL,
        target_distance_of_measured_SPL,
        distance,
        noise_floor_SPL) for distance in distances]

    print(norm_values_in_range)














