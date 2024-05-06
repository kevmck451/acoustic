



from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af


import numpy as np

def calculate_RMS(audio):
    """Calculate the power of an audio signal."""
    return np.sqrt(np.sum(audio ** 2) / len(audio))
    # return np.sum(audio ** 2) / len(audio)

def find_norm_value(value, noise, target):

    ratio_found = False
    noise_norm_value = value
    target_norm_value = value
    precision_value = 6
    while not ratio_found:
        normalized_noise = process.normalize(noise, percentage=noise_norm_value)
        normalized_target = process.normalize(target, percentage=target_norm_value)

        noise_power = np.round(calculate_RMS(normalized_noise.data), precision_value)
        target_power = np.round(calculate_RMS(normalized_target.data), precision_value)

        if noise_power == target_power:
            ratio_found = True
        else:
            if noise_power > target_power:
                target_norm_value += 0.1
            else:
                noise_norm_value -= 0.1

        # print(f'Noise RMS: {noise_power} / Target RMS: {target_power} / T_Norm: {target_norm_value}')
    norm_difference = target_norm_value - noise_norm_value

    print(value, target_norm_value)
    return norm_difference

def find_noise_target_norm_relationship(noise, target):
    norm_list = [x for x in range(1, 101)]
    values_list = [find_norm_value(x, noise, target) for x in norm_list]

    # Perform linear regression: deg=1 for linear
    slope, intercept = np.polyfit(norm_list, values_list, deg=1)

    # Create a linear model function based on the slope and intercept
    linear_model = np.poly1d([slope, intercept])

    # Generate y values based on the model to plot or analyze
    y_model = linear_model(norm_list)
    # y_model represents how much greater the target norm value is to noise norm value on full norm scale
    import matplotlib.pyplot as plt
    plt.plot(norm_list, values_list)
    plt.plot(norm_list, y_model)
    plt.show()

    return linear_model

def get_target_value_based_on_noise_value_for_one_to_one(noise_val, noise, target):
    # this is for 1:1 SNR
    linear_function = find_noise_target_norm_relationship(noise, target)
    addition = linear_function(noise_val)
    tar_val = noise_val + addition
    return np.round(tar_val, 3)

def square_law_equation(measured_SPL, distance_of_measured_SPL, input_distance):
    # measured_SPL is measured in dB(C)
    # distance_of_measured_SPL and input_distance in meters

    desired_SPL = np.round((measured_SPL - 20 * (np.log10(input_distance / distance_of_measured_SPL))), 5)

    return desired_SPL

# def norm_value_from_distance(target_measured_SPL, target_distance_of_measured_SPL, distance, noise_floor_SPL, target_norm_val):
#     distance_SPL = square_law_equation(
#         target_measured_SPL,
#         target_distance_of_measured_SPL,
#         distance)
#
#     percentage_of_SPL = (distance_SPL / noise_floor_SPL)
#     norm_value = target_norm_val * percentage_of_SPL
#     return norm_value

def convert_SPL_to_RMS(SPL_value):
    """
        Convert SPL value to RMS sound pressure.

        Parameters:
        - spl: Sound Pressure Level in decibels (dB).
        - p_ref: Reference sound pressure in Pascals (Pa), default is 20 µPa.

        Returns:
        - RMS sound pressure in Pascals (Pa).
        """
    p_ref = 2e-5
    rms_pressure = p_ref * np.power(10, SPL_value / 20)
    return rms_pressure

def find_norm_value_from_rms(target, RMS_desired):
    ratio_found = False
    target_norm_value = 80
    precision_value = 2
    increment_value = 1/(10**precision_value-1)


    while not ratio_found:
        normalized_target = process.normalize(target, percentage=target_norm_value)

        target_power = np.round(calculate_RMS(normalized_target.data), precision_value)
        rms_compare = np.round(RMS_desired, precision_value)

        if rms_compare == target_power:
            print('here')
            ratio_found = True
        else:
            if rms_compare > target_power:
                target_norm_value += increment_value
            else:
                target_norm_value -= increment_value

        print(f'RMS_D: {rms_compare} / RMS: {target_power} / T_Norm: {target_norm_value}')
    return target_norm_value

def generate_synthetic_dataset_distance(noise_path, target_path,
                                        sample_rate, sample_length,
                                        target_measured_SPL, target_distance_of_measured_SPL,
                                        noise_floor_SPL, distances_of_interest):


    # Get list of SPL values at distances
    distances = [x for x in np.arange(10, 101, 1)]
    target_SPL_values = [square_law_equation(
        target_measured_SPL,
        target_distance_of_measured_SPL,
        distance) for distance in distances]

    # List of Target's SPL values by Distances
    # Convert SPL values to RMS values
    # RMS_values = [convert_SPL_to_RMS(SPL) for SPL in target_SPL_values]
    # print(RMS_values)

    # Target Norm Value = 95.071 -> Related to 98 dB(C) SPL Measurement -> Related to 6.7 meters from target

    # I want to know what the target norm value would be at X meters

    # Relate to Normalize Values
    noise_sample = Audio_Abstract(filepath=noise_path, sample_rate=sample_rate)
    target_sample = Audio_Abstract(filepath=target_path, sample_rate=sample_rate)
    noise_chunk_list, _ = process.generate_chunks(noise_sample, length=sample_length)
    target_chunk_list, _ = process.generate_chunks(target_sample, length=sample_length)
    noise = noise_chunk_list[0]
    target = target_chunk_list[0]

    # norm_value_list = [find_norm_value_from_rms(target, RMS_desired) for RMS_desired in RMS_values]
    # print(norm_value_list)
    # import matplotlib.pyplot as plt
    # plt.plot(distances, target_SPL_values, label='SPL')
    # plt.plot(distances, RMS_values, label='RMS')
    # plt.plot(distances, norm_value_list, label='Norm')
    # plt.legend()
    # plt.show()


    noise_norm_val = 90  # arbitrary value
    target_norm_val = get_target_value_based_on_noise_value_for_one_to_one(noise_norm_val, noise, target)
    print(f'Noise Norm Val: {noise_norm_val} / Tar Norm Val: {target_norm_val}')

    # Find Distance Where UAV and Target are Equal
    distance_equals_index = min(enumerate(target_SPL_values), key=lambda x: abs(x[1] - noise_floor_SPL))[0]
    distance_where_equal = np.round(distances[distance_equals_index], 5)
    print(f'Distance where Equal: {distance_where_equal} meters')











if __name__ == '__main__':

    # Input Sample Info
    noise_path = af.hex_hover_combo_thick
    target_path = af.diesel_tank_1_3
    sample_rate = 24_000
    sample_length = 10

    # UAV Info
    noise_floor_SPL = 98

    # Target Info
    target_measured_SPL = 93  # dB(C)
    target_distance_of_measured_SPL = 12  # m

    distances_of_interest = [x for x in range(30, 50, 1)]

    # Generate Dataset
    generate_synthetic_dataset_distance(noise_path, target_path, sample_rate, sample_length,
                                        target_measured_SPL, target_distance_of_measured_SPL, noise_floor_SPL,
                                        distances_of_interest)




















