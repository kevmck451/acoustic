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
from Acoustic.audio_multich import Audio_MC
from scipy import signal







# Function to calculate POWER
def power_types(filepath):
    # RMS
    # Average Power
    # Power Spectral Density

    audio_object = Audio_MC(filepath)

    # Max value for 16 bit data: 32767
    rms = (np.sqrt(np.mean(np.square(audio_object.data))) / 32767).round(3)
    average_power = (np.mean(np.abs(audio_object.data))/ 32767).round(3)
    frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate)

    print(f'RMS: {rms}\nAP: {average_power}')

    # Plot the Power Spectral Density
    plt.semilogy(frequencies, psd[1])
    plt.title('Power Spectral Density of Audio Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.show()











def signal_noise_ratio(signal_obejct):
    pass


# Function to compute signal to noise for 3 signals: signal, noise, sig+noise
def signal_noise_ratio_3(signal, noise, noisy_signal, econ, display=True):
    signal_ob = Audio_MC(signal)
    noisy_signal_ob = Audio_MC(noisy_signal)
    noise_ob = Audio_MC(noise)

    signal_data = adjust_signal(signal_ob, econ[0], econ[1], econ[2], econ[3], econ[4], econ[5])

    rate = signal_ob.sample_rate
    # rate = 20000

    # Calculating average powers for all channels
    signal_power = np.mean(np.square(signal_data), axis=1)
    noise_power = np.mean(np.square(noise_ob.data), axis=1)
    noisy_signal_power = np.mean(np.square(noisy_signal_ob.data), axis=1)

    # Estimating the signal power at the noisy signal location
    estimated_signal_power = noisy_signal_power - noise_power
    estimated_signal_power = np.clip(estimated_signal_power, 1e-10, None)

    # Calculating SNR
    snr = 10 * np.log10(estimated_signal_power / noise_power)
    average_snr = np.mean(snr)

    if display:
        # Create plots
        plt.figure(figsize=(14, 8))

        # Spectrograms
        plt.subplot(2, 3, 1)
        plt.specgram(signal_data.mean(axis=0), Fs=rate)
        plt.title("Signal Spectrogram")

        plt.subplot(2, 3, 2)
        plt.specgram(noisy_signal_ob.data.mean(axis=0), Fs=rate)
        plt.title("Noisy Signal Spectrogram")

        plt.subplot(2, 3, 3)
        plt.specgram(noise_ob.data.mean(axis=0), Fs=rate)
        plt.title("Noise Spectrogram")

        # Histograms
        plt.subplot(2, 3, 4)
        plt.hist(signal_power, bins=50)
        plt.title("Signal Power Histogram")
        plt.xlabel("Power")
        plt.ylabel("Frequency")

        plt.subplot(2, 3, 5)
        plt.hist(noisy_signal_power, bins=50)
        plt.title("Noisy Signal Power Histogram")
        plt.xlabel("Power")
        plt.ylabel("Frequency")

        plt.subplot(2, 3, 6)
        plt.hist(noise_power, bins=50)
        plt.title("Noise Power Histogram")
        plt.xlabel("Power")
        plt.ylabel("Frequency")

        plt.tight_layout(pad=1)
        plt.show()

    # Print SNR
    print(f"The average SNR for the system is {average_snr:.2f} dB.")

    return average_snr

# More Complex Function
def adjust_signal(signal, original_distance, reference_distance, freq, temperature, humidity, pressure):

    # Calculate total absorption, speed of sound and air density
    abs_coeff_db, sound_speed, air_dens = calc_coeff(freq, original_distance, temperature, humidity, pressure)

    # Calculate acoustic intensity from pressure of original signal using I = P^2 / (ρc)
    signal_intensity = np.power(signal.data, 2) / (air_dens * sound_speed)

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

# Function to calculate several key properties related to sound propagation
def calc_coeff(freq, distance, temperature, humidity, pressure):
    pres = pressure * 100  # convert hPa to Pa
    p_ref = 20e-6  # 20 μPa = 20e-6 Pa
    temp = (temperature - 32) * 5 / 9 + 273.15
    # Calculate the saturation vapor pressure using the Antoine equation
    p_sat_ref = np.power(10, -6.8346 * np.power(273.16 / temp, 1.261) + 4.6151)
    # Calculate the molar concentration of water vapor using the relative humidity, saturation pressure, barometric pressure, and reference pressure
    mol_conc_wv = (100 * humidity * (p_sat_ref / (pres / p_ref))) / 100
    # Calculate the oxygen relaxation frequency in the air-water vapor mixture
    mol_conc_water_vapor = 100 * mol_conc_wv
    oxy_freq = (pres / p_ref) * (24 + 40400 * mol_conc_water_vapor * ((0.02 + mol_conc_water_vapor) / (0.391 + mol_conc_water_vapor)))
    # Calculate the nitrogen relaxation frequency in the air-water vapor mixture
    mol_conc_water_vapor = 100 * mol_conc_wv
    nit_freq = (pres / p_ref) * np.power(temp / 293.15, -0.5) * (9 + 280 * mol_conc_water_vapor * np.exp(-4.17 * (np.power(temp / 293.15, -1 / 3) - 1)))
    # Calculate the absorption coefficient of sound in the air-water vapor mixture and multiply it by the distance to get the total absorption
    absorption_coeff = 10*np.log10(np.exp(np.power(freq, 2) * (1.84 * (10**-11) *
                       np.power(pres / p_ref, -1) *
                       np.power(temp / 293.15, 1 / 2) + np.power(temp / 293.15, -5 / 2) * (0.01275 *
                       np.exp(-2239 / temp) * (oxy_freq /
                       (np.power(freq, 2) + np.power(oxy_freq, 2))) + 0.1068 * np.exp(-3352 / temp) *
                       (nit_freq / (np.power(freq, 2) + np.power(nit_freq, 2)))))))
    abs_coeff_db = distance * absorption_coeff

    # Calculate the molar mass of the air-water vapor mixture
    mol_mix = mol_conc_wv * 0.018016 + (1 - mol_conc_wv) * 0.02897
    # Calculate the heat capacity ratio (gamma) for the air-water vapor mixture
    hcr_mix = 1 / (mol_conc_wv / (1.33 - 1) + (1 - mol_conc_wv) / (1.4 - 1)) + 1
    # Calculate the speed of sound in the air-water vapor mixture
    sound_speed = np.sqrt(hcr_mix * 8.314462 * temp / mol_mix)
    # Calculate the density of the air-water vapor mixture
    air_dens = mol_mix * pres / (8.314462 * temp)

    # Return the total absorption, speed of sound and air density
    return abs_coeff_db, sound_speed, air_dens










