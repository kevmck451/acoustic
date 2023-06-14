# Comparing SNR for tones / engines from studio monitor and multirotor drone

'''
Tone Signals @ 2 meters:
    - 250 Hz
    - 400 Hz
    - 750 Hz
    - 1000 Hz

Noise:
    - Hover at 10m, 20m, 30m, 40m

Tone Noisy Signal:
    - 250 Hz, 400 Hz, 750 Hz, 1000 Hz @ 10 meters
    - 250 Hz, 400 Hz, 750 Hz, 1000 Hz @ 20 meters
    - 250 Hz, 400 Hz, 750 Hz, 1000 Hz @ 30 meters
    - 250 Hz, 400 Hz, 750 Hz, 1000 Hz @ 40 meters
'''

'''
Please note that the signal-to-noise ratio (SNR) 
calculation requires determining the power of the 
signal and the noise. Power is usually calculated 
as the average of the square of the signal, and the 
SNR is generally given in decibels, calculated as 
10 * log10(signal power / noise power).

The distance adjustment is a little more complex. 
If we know the relation of signal attenuation with 
distance, we can adjust the signal to a reference 
distance. In the following example, I assume a simple 
inverse law (which is oversimplified for real-world 
acoustics, but is used for demonstration).

'''

import numpy as np
import matplotlib.pyplot as plt
# from environment import *


def signal_noise_ratio(signal, noisy_signal, noise, original_distance, reference_distance=1):
    # Adjust signal for distance
    signal_adjusted = adjust_signal_1(signal, original_distance, reference_distance)
    signal_adjusted = adjust_signal_2(signal, original_distance, reference_distance, freqs, temperature, rel_hum, p_bar, p_ref)

    # We calculate SNR for each channel separately
    snr_list = []
    for i in range(4):
        # Calculate power
        signal_power = np.mean(np.square(signal_adjusted.data[i]))
        noise_power = np.mean(np.square(noise.data[i]))
        noisy_signal_power = np.mean(np.square(noisy_signal.data[i]))

        # We use the noisy signal to estimate the actual signal power at the noisy location
        estimated_signal_power = noisy_signal_power - noise_power

        # If estimated signal power turns out negative, it's likely due to noise power over-estimation
        # We clip it to a small positive value to avoid undefined dB calculations
        estimated_signal_power = np.clip(estimated_signal_power, 1e-10, None)

        # Calculate SNR in dB for this channel
        snr = 10 * np.log10(estimated_signal_power / noise_power)
        snr_list.append(snr)

    # Plot SNR for each channel
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 5), snr_list, tick_label=['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4'])
    plt.xlabel('Channel')
    plt.ylabel('Signal-to-Noise Ratio (dB)')
    plt.title('Signal-to-Noise Ratio for each Channel')
    plt.grid(True)
    plt.show()

    # Return average SNR over all channels
    return np.mean(snr_list)


# Basic Function
def adjust_signal_1(signal, original_distance, reference_distance=1):
    return signal * (original_distance / reference_distance)

# More Complex Function
def adjust_signal_2(signal, original_distance, reference_distance, freqs, temperature, rel_hum, p_bar, p_ref):
    # Calculate total absorption, speed of sound and air density
    abs_coeff_db, sound_speed, air_dens = calc_coeff(freqs, original_distance, temperature, rel_hum, p_bar, p_ref)

    # Calculate acoustic intensity from pressure of original signal using I = P^2 / (ρc)
    signal_intensity = np.power(signal, 2) / (air_dens * sound_speed)

    # Calculate sound power from acoustic intensity using P = I*4πr^2 / Q
    # We assume directivity factor (Q) is 1 for simplicity
    signal_power = signal_intensity * 4 * np.pi * np.square(original_distance) / 1

    # Calculate reference acoustic intensity from sound power using I = P*Q / (4πr^2)
    ref_intensity = signal_power * 1 / (4 * np.pi * np.square(reference_distance))

    # Calculate reference pressure from acoustic intensity using P = sqrt(Iρc)
    signal_adjusted = np.sqrt(ref_intensity * air_dens * sound_speed)

    return signal_adjusted


'''This function, calc_coeff, seems to be an overall function that calculates 
several key properties related to sound propagation in an air-water vapor mixture, 
given inputs such as frequency, distance, temperature, relative humidity, 
barometric pressure, and reference pressure. These properties include the 
total absorption of sound (in decibels), the speed of sound in the mixture, 
and the density of the mixture. '''


def calc_coeff(freq, distance, temperature, rel_hum, p_bar, p_ref):
    # Calculate the saturation vapor pressure using the Antoine equation
    p_sat_ref = np.power(10, -6.8346*np.power(273.16/temperature, 1.261) + 4.6151)
    # Calculate the molar concentration of water vapor using the relative humidity, saturation pressure, barometric pressure, and reference pressure
    mol_conc_wv = (100*rel_hum*(p_sat_ref/(p_bar/p_ref)))/100
    # Calculate the oxygen relaxation frequency in the air-water vapor mixture
    mol_conc_water_vapor = 100 * mol_conc_wv
    oxy_freq = (p_bar/p_ref)*(24 + 40400*mol_conc_water_vapor*((0.02 + mol_conc_water_vapor)/(0.391 + mol_conc_water_vapor)))
    # Calculate the nitrogen relaxation frequency in the air-water vapor mixture
    mol_conc_water_vapor = 100 * mol_conc_wv
    nit_freq = (p_bar/p_ref)*np.power(temperature/293.15, -0.5)*(9 + 280*mol_conc_water_vapor*np.exp(-4.17*(np.power(temperature/293.15, -1/3) - 1)))
    # Calculate the absorption coefficient of sound in the air-water vapor mixture and multiply it by the distance to get the total absorption
    absorption_coeff = 10*np.log10(np.exp(np.power(freq, 2) * (1.84 * (10**-11) *
                        np.power(p_bar/p_ref, -1) *
                        np.power(temperature/293.15, 1/2) + np.power(temperature/293.15, -5/2) * (0.01275 *
                        np.exp(-2239/temperature) * (oxy_freq /
                        (np.power(freq, 2) + np.power(oxy_freq, 2))) + 0.1068 * np.exp(-3352 / temperature) *
                        (nit_freq / (np.power(freq, 2) + np.power(nit_freq, 2)))))))
    abs_coeff_db = distance * absorption_coeff

    # Calculate the molar mass of the air-water vapor mixture
    mol_mix = mol_conc_wv * 0.018016 + (1 - mol_conc_wv) * 0.02897
    # Calculate the heat capacity ratio (gamma) for the air-water vapor mixture
    hcr_mix = 1 / (mol_conc_wv / (1.33 - 1) + (1 - mol_conc_wv) / (1.4 - 1)) + 1
    # Calculate the speed of sound in the air-water vapor mixture
    sound_speed = np.sqrt(hcr_mix * 8.314462 * temperature / mol_mix)
    # Calculate the density of the air-water vapor mixture
    air_dens = mol_mix * p_bar / (8.314462 * temperature)

    # Return the total absorption, speed of sound and air density
    return abs_coeff_db, sound_speed, air_dens










