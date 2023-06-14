import numpy as np
# from environ_constants import *
from scipy import signal
import pandas as pd

# Constants:

# Mic hole
mic_rad_inch = 0.03125
mic_rad_mm = mic_rad_inch*25.4
cs_area = np.pi*np.square(mic_rad_mm/1000)

# SNR
window = 50

# Matched Filter
mat_filt_welch = [4, 3999]
big_sr = 48000

dst_from_src = 1











# Perform Welch's power spectral density estimate on both datasets
def get_SNR_arrays(data_1, data_2, snr_type):
    '''
    This function calculates the Signal-to-Noise Ratio (SNR) between two data sets 
    (either representing two different signals, or one representing a signal and 
    the other representing noise). The type of SNR calculation is determined by 
    the snr_type parameter, which could represent a given signal, given noise, 
    a system measurement, or a pure signal-to-noise ratio calculation. The SNR 
    is calculated for each frequency and can be smoothed by applying a rolling window. 
    The function returns the frequency axis and the SNR in different forms: in raw decibels, 
    decibels after initial smoothing, and decibels after a second round of smoothing. 
    This allows for various interpretations and uses of the SNR data in further analysis or visualization.
    '''


    # Welch's method is used to estimate the power of a signal at different frequencies
    data_1_freq, data_1_data = signal.welch(x=data_1, fs=big_sr, nperseg=32768, average='mean')
    data_2_freq, data_2_data = signal.welch(x=data_2, fs=big_sr, nperseg=32768, average='mean')
    
    # Apply a rolling window to the power spectral density estimate to smooth the data
    data_1_roll = pd.Series(data_1_data).rolling(window, center=True).mean().to_numpy()
    data_2_roll = pd.Series(data_2_data).rolling(window, center=True).mean().to_numpy()
    
    # If the SNR type is 'System', subtract the first dataset's power spectral density from the second's
    if snr_type == 'System':
        data_2_data = data_2_data - data_1_data
        data_2_roll = data_2_roll - data_1_roll
    
    # Initialize lists to hold the Signal-to-Noise Ratio (SNR) values
    snr_plain = []
    db_plain = []
    snr_rolled_before = []
    db_rolled_before = []
    # Calculate SNR according to the method specified by 'snr_type'
    for l1, l2, l1r, l2r in zip(data_1_data, data_2_data, data_1_roll, data_2_roll):
        if snr_type == 'Given Signal':
            # If the signal is given, calculate the SNR using the formula: 1/((noisy_signal/noise)-1)
            this_data_ratio = 1/(((l2*1.25)/l1) - 1)
            this_roll_ratio = 1/(((l2r*1.25)/l1r) - 1)
        elif snr_type == 'Given Noise':
            # If the noise is given, calculate the SNR using the formula: (signal/noise)-1
            this_data_ratio = ((l1)/l2) - 1
            this_roll_ratio = ((l1r)/l2r) - 1
        else:
            # If 'snr_type' is neither 'Given Signal' nor 'Given Noise', calculate the SNR as the ratio of the two inputs
            this_data_ratio = l1/l2
            this_roll_ratio = l1r/l2r
        # Append the SNR to the respective lists in raw form and in decibels (dB)
        snr_plain.append(this_data_ratio)
        db_plain.append(10*np.log10(this_data_ratio))
        snr_rolled_before.append(this_roll_ratio)
        db_rolled_before.append(10*np.log10(this_roll_ratio))
    
    # Apply a rolling window to the calculated SNR (in dB) to smooth the data
    db_rolled_both = 10*np.log10(pd.Series(snr_rolled_before).rolling(window, center=True).mean().to_numpy())
    
    # Return the frequency axis, and the SNR in different forms: raw dB, dB after first smoothing, and dB after second smoothing
    return data_1_freq, db_plain, db_rolled_before, db_rolled_both


'''
This function calculates the predicted signal-to-noise ratio (SNR) for a set of frequencies given 
various parameters of the environment and the sound source. The SNR is a measure that compares the 
level of a desired signal to the level of background noise. The process involves calculating the 
signal and noise powers at the receiver, converting them to decibels, and then computing the 
difference between them to obtain the SNR.
'''

# Function to calculate the predicted signal-to-noise ratio (SNR) for given frequencies, source sound pressure level, directivity factor, velocity array, distance, temperature, relative humidity, barometric pressure, and reference pressure
def calc_snr_pred(freqs, src_spl, dir_fact, vel_array, distance, temperature, rel_hum, p_bar, p_ref):
    # Calculate absorption coefficient, speed of sound, and air density for given frequencies, temperature, relative humidity, barometric pressure, and reference pressure
    abs_coeff_db, sound_speed_const, air_dens = calc_coeff(freqs, 1, temperature, rel_hum, p_bar, p_ref)

    # Calculate the speed of sound with wind 
    sos_wind = sound_speed_const + vel_array[1]
    
    # Convert the source sound pressure level to acoustic pressure
    src_p_acc = value_db_conv(src_spl, 'value', 'pressure', 'value')

    # Convert the source acoustic pressure to intensity
    src_intensity = p_acc_intensity_conv(src_p_acc, 'pressure', sos_wind, air_dens)

    # Convert the source intensity to power
    src_pwr = pwr_intensity_conv(src_intensity, 'intensity', dst_from_src, dir_fact)

    # Convert the source power to dB
    src_pwr_db = value_db_conv(src_pwr, 'value', 'power', 'db')

    # Create an empty array to hold the sound pressure level at the receiver
    src_spl_from_dist = np.empty(shape=len(freqs))

    # Calculate the sound pressure level at the receiver for each frequency
    for i in range(len(src_spl_from_dist)):
        src_spl_from_dist[i] = src_pwr_db - 10 * np.log10(4 * np.pi * (distance ** 2) / dir_fact) - abs_coeff_db[i] * distance

    # Convert the sound pressure level at the receiver to acoustic pressure
    src_p_acc_dist = value_db_conv(src_spl_from_dist, 'value', 'pressure', 'value')

    # Convert the acoustic pressure at the receiver to intensity
    src_int_dist = p_acc_intensity_conv(src_p_acc_dist, 'pressure', sos_wind, air_dens)

    # Convert the intensity at the receiver to power
    src_pwr_dist = pwr_intensity_conv(src_int_dist, 'intensity', 1, dir_fact)

    # Convert the power at the receiver to dB
    src_pwr_db_dist = value_db_conv(src_pwr_dist, 'value', 'power', 'db')

    # Compute the average dB power at the receiver
    src_pwr_db_dist = db_array_to_mean(src_pwr_db_dist)

    # Compute the noise power in dB 
    pow_wind = 0.5 * air_dens * cs_area * np.power(vel_array[0] + vel_array[2], 3)
    noise_pwr_db = 10 * np.log10(pow_wind / (10 ** (-12)))

    # Return the predicted SNR in dB 
    return src_pwr_db_dist - noise_pwr_db

# Function to convert between decibel and linear values, taking into account the type of value and whether it's a value or a ratio
def value_db_conv(val, val_rat, val_type, result_type, ref=10**-12):
    factor = 10
    # If the value type is pressure, voltage, or current, adjust the conversion factor and reference value
    if val_type == 'pressure' or val_type == 'voltage' or val_type == 'current':
        factor = 20
        ref = 2*(10**-5) if val_type == 'pressure' else ref
    # Depending on the result type and whether the input is a value or a ratio, convert the input accordingly
    if result_type == 'value' and val_rat == 'value':
        return np.power(10, (val/factor))*ref
    elif result_type == 'value' and val_rat == 'ratio':
        return np.power(10, (val/factor))
    elif result_type == 'db' and val_rat == 'value':
        return factor*np.log10(val/ref)
    else: # result_type == 'db' and val_rat == 'ratio'
        return factor*np.log10(val)

# Function to calculate the mean value of an array of decibel values
def db_array_to_mean(db_array):
    avg_ratio = np.mean(value_db_conv(db_array, 'ratio', 'power', 'value'))
    return value_db_conv(avg_ratio, 'ratio', 'power', 'db')

# Function to calculate the average signal-to-noise ratio (SNR) in decibels for an array of distances
def find_avg_snr_db_dist_array(dist_array, snr_db=None, vel_array=angel_vel_array, special_dist=angel_alt):
    if not type(dist_array) == np.ndarray:
        dist_array = [dist_array]
    snr_pred_db = calc_snr_pred(freqs, spl_src, dir_fact, vel_array, special_dist, angel_temp, angel_rel_hum, angel_p_bar, angel_p_ref)
    diff = db_array_to_mean(snr_pred_db) - snr_db if snr_db else 0

    snr_avg_db_dist_model = []
    snr_avg_db_dist = []
    for dist in dist_array:
        snr_avg_db_model = db_array_to_mean(calc_snr_pred(freqs, spl_src, dir_fact, vel_array, dist, angel_temp, angel_rel_hum, angel_p_bar, angel_p_ref))
        snr_avg_db_dist_model.append(snr_avg_db_model)
        snr_avg_db_dist.append(snr_avg_db_model - diff)
    if diff == 0:
        snr_avg_db_dist = None
    return dist_array, snr_avg_db_dist, snr_avg_db_dist_model

# Function to convert between pressure and acoustic intensity
def p_acc_intensity_conv(signal, in_type, sound_speed, air_dens):
    # If the input type is pressure
    if in_type == 'pressure':
        # Use the formula I = P^2 / (ρc) to convert pressure to acoustic intensity
        return np.power(signal, 2) / (air_dens * sound_speed)
    else: # in_type == 'intensity'
        # Use the formula P = sqrt(Iρc) to convert acoustic intensity to pressure
        return np.sqrt(signal * air_dens * sound_speed)

# Function to convert between sound power and sound intensity
def pwr_intensity_conv(val, in_type, distance, dir_fact):
    # If the input type is power
    if in_type == 'power':
        # Use the formula I = P*Q / (4πr^2) to convert sound power to sound intensity
        return val*dir_fact/(4*np.pi*np.square(distance))
    else: # in_type == 'intensity'
        # Use the formula P = I*4πr^2 / Q to convert sound intensity to sound power
        return val*4*np.pi*np.square(distance)/dir_fact



'''This function, calc_coeff, seems to be an overall function that calculates 
several key properties related to sound propagation in an air-water vapor mixture, 
given inputs such as frequency, distance, temperature, relative humidity, 
barometric pressure, and reference pressure. These properties include the 
total absorption of sound (in decibels), the speed of sound in the mixture, 
and the density of the mixture. '''
def calc_coeff(freqs, distance, temperature, rel_hum, p_bar, p_ref):
    # Calculate the saturation vapor pressure using the Antoine equation
    p_sat_ref = p_sat_ref_easy(temperature)
    # Calculate the molar concentration of water vapor using the relative humidity, saturation pressure, barometric pressure, and reference pressure
    mol_conc_wv = mol_conc_water_vapor(rel_hum, p_sat_ref, p_bar, p_ref)
    # Calculate the oxygen relaxation frequency in the air-water vapor mixture
    oxy_freq = oxy_relax_freq(p_bar, p_ref, 100*mol_conc_wv)
    # Calculate the nitrogen relaxation frequency in the air-water vapor mixture
    nit_freq = nit_relax_freq(temperature, p_bar, p_ref, 100*mol_conc_wv)
    # Calculate the absorption coefficient of sound in the air-water vapor mixture and multiply it by the distance to get the total absorption
    abs_coeff_db = distance*absorption_coeff(temperature, p_bar, p_ref, freqs, oxy_freq, nit_freq)
    
    # Calculate the molar mass of the air-water vapor mixture
    mol_mix = mol_mass_mix(mol_conc_wv)
    # Calculate the heat capacity ratio (gamma) for the air-water vapor mixture
    hcr_mix = heat_cap_ratio_mix(mol_conc_wv)
    # Calculate the speed of sound in the air-water vapor mixture
    sound_speed = speed_of_sound(temperature, mol_mix, hcr_mix)
    # Calculate the density of the air-water vapor mixture
    air_dens = air_density(temperature, p_bar, mol_mix)
    
    # Return the total absorption, speed of sound and air density
    return abs_coeff_db, sound_speed, air_dens

# Function to calculate saturation vapor pressure using Antoine equation
def p_sat_ref_easy(temperature):
    return np.power(10, -6.8346*np.power(273.16/temperature, 1.261) + 4.6151)

# Function to calculate molar concentration of water vapor using the relative humidity, saturation pressure, barometric pressure and reference pressure
def mol_conc_water_vapor(rel_hum, p_sat_ref, p_bar, p_ref):
    return (100*rel_hum*(p_sat_ref/(p_bar/p_ref)))/100

# Function to calculate the molar mass of the air-water vapor mixture
def mol_mass_mix(mol_conc_wv):
    return mol_conc_wv * 0.018016 + (1 - mol_conc_wv) * 0.02897

# Function to calculate the heat capacity ratio (gamma) for the air-water vapor mixture
def heat_cap_ratio_mix(mol_conc_wv):
    return 1 / (mol_conc_wv / (1.33 - 1) + (1 - mol_conc_wv) / (1.4 - 1)) + 1

# Function to calculate the speed of sound in the air-water vapor mixture
def speed_of_sound(temperature, mol_mix, hcr_mix):
    return np.sqrt(hcr_mix * 8.314462 * temperature / mol_mix)

# Function to calculate the density of the air-water vapor mixture
def air_density(temperature, p_bar, mol_mix):
    return mol_mix * p_bar / (8.314462 * temperature)

# Function to calculate the oxygen relaxation frequency in the air-water vapor mixture
def oxy_relax_freq(p_bar, p_ref, mol_conc_water_vapor):
    return (p_bar/p_ref)*(24 + 40400*mol_conc_water_vapor*((0.02 + mol_conc_water_vapor)/(0.391 + mol_conc_water_vapor)))

# Function to calculate the nitrogen relaxation frequency in the air-water vapor mixture
def nit_relax_freq(temperature, p_bar, p_ref, mol_conc_water_vapor):
    return (p_bar/p_ref)*np.power(temperature/293.15, -0.5)*(9 + 280*mol_conc_water_vapor*np.exp(-4.17*(np.power(temperature/293.15, -1/3) - 1)))

# Function to calculate the absorption coefficient of sound in the air-water vapor mixture
def absorption_coeff(temperature, p_bar, p_ref, freq, oxy_freq, nit_freq):
    return 10*np.log10(np.exp(np.power(freq, 2) * (1.84 * (10**-11) *
                                                   np.power(p_bar/p_ref, -1) *
                                                   np.power(temperature/293.15, 1/2) + np.power(temperature/293.15, -5/2) * (0.01275 *
                                                                                                                             np.exp(-2239/temperature) * (oxy_freq /
                                                                                                                                     (np.power(freq, 2) + np.power(oxy_freq, 2))) + 0.1068 * np.exp(-3352 / temperature) *
                                                                                                                             (nit_freq / (np.power(freq, 2) + np.power(nit_freq, 2)))))))

















