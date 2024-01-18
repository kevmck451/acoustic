from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import skew
import scipy.stats
from scipy.stats import norm







def hex_hover_stats(**kwargs):
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    harmonics = kwargs.get('harmonics', [(95, 117), (190, 224), (294, 336), (395, 445), (495, 540)])

    # Initialize a dictionary to accumulate stats for each harmonic
    harmonic_stats = {i: {'power_sum': 0, 'count': 0} for i in range(1, len(harmonics) + 1)}

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, norm=True)

        # Calculate the power within each harmonic range and accumulate it
        for i, (left_freq, right_freq) in enumerate(harmonics, start=1):
            # Find the indices of the frequency bins within the harmonic range
            indices = np.where((f_bins >= left_freq) & (f_bins <= right_freq))[0]

            # Sum the power within this range
            power = np.sum(spectrum[indices])

            # Update the harmonic stats
            harmonic_stats[i]['power_sum'] += power
            harmonic_stats[i]['count'] += len(indices)

    # Calculate the average PSD for each harmonic range
    for i in harmonic_stats:
        if harmonic_stats[i]['count'] > 0:
            harmonic_stats[i]['average_psd'] = harmonic_stats[i]['power_sum'] / harmonic_stats[i]['count']
        else:
            harmonic_stats[i]['average_psd'] = 0

    print(harmonic_stats)
    return harmonic_stats


def spectral_subtraction(spectrum, f_bins, harmonic_stats, **kwargs):
    harmonics = kwargs.get('harmonics', [(95, 117), (190, 224), (294, 336), (395, 445), (495, 540)])

    # Create a copy of the spectrum to hold the noise-reduced version
    clean_spectrum = np.copy(spectrum)

    # Subtract the average PSD from the spectrum at the harmonic ranges
    for harmonic, stats in harmonic_stats.items():
        left_freq, right_freq = harmonics[harmonic - 1]  # Get the frequency range for this harmonic
        within_range = (f_bins >= left_freq) & (f_bins <= right_freq)
        clean_spectrum[within_range] -= stats['average_psd']

        # Ensure no negative values after subtraction
        clean_spectrum[within_range] = np.maximum(clean_spectrum[within_range], 0)

    display = kwargs.get('display', False)
    if display:
        # plt.plot(frequency_bins, average_spectrum)
        plt.plot(f_bins[:10000], clean_spectrum[:10000])
        plt.xlabel('Frequency (Hz)', fontweight='bold')
        plt.ylabel('Magnitude', fontweight='bold')
        plt.title(f'Spectral Plot')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout(pad=1)

        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/something')
            plt.close()
        else:
            plt.show()

    clean_spectrum[clean_spectrum < 0] = 0
    # print(np.min(clean_spectrum))

    return clean_spectrum


def spectral_subtraction_2(spectrum, f_bins, harmonic_stats, **kwargs):
    harmonics = kwargs.get('harmonics', [(95, 117), (190, 224), (294, 336), (395, 445), (495, 540)])
    clean_spectrum = np.copy(spectrum)

    # Loop over each harmonic range
    for harmonic, stats in harmonic_stats.items():
        left_freq, right_freq = harmonics[harmonic - 1]  # Get the frequency range for this harmonic
        within_range = (f_bins >= left_freq) & (f_bins <= right_freq)

        # Estimate the noise spectrum using the average PSD as the mean of a Gaussian distribution
        # The noise magnitude for each frequency bin is drawn from this distribution
        noise_mean = stats['average_psd']
        noise_std = np.sqrt(stats.get('power_variance', noise_mean))  # If variance is not known, use mean
        noise_pdf = norm(noise_mean, noise_std).pdf(f_bins[within_range])

        # Subtract the estimated noise from the spectrum
        clean_spectrum[within_range] -= noise_pdf

        # Ensure no negative values after subtraction
        clean_spectrum[within_range] = np.maximum(clean_spectrum[within_range], 0)

    display = kwargs.get('display', False)
    if display:
        # plt.plot(frequency_bins, average_spectrum)
        plt.plot(f_bins[:10000], clean_spectrum[:10000])
        plt.xlabel('Frequency (Hz)', fontweight='bold')
        plt.ylabel('Magnitude', fontweight='bold')
        plt.title(f'Spectral Plot')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout(pad=1)

        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/something')
            plt.close()
        else:
            plt.show()

    clean_spectrum[clean_spectrum < 0] = 0

    return clean_spectrum


# Function to see average spectrum for hex hovering
def hex_hover_average_spectrum():
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    x_range = [80, 1000]
    min_y, max_y = float('inf'), float('-inf')
    fig, ax = plt.subplots(figsize=(8, 4))

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, norm=True)

        ax.plot(f_bins, spectrum, alpha=0.25)

        # Calculate Stats for Viewing
        within_range = (f_bins >= x_range[0]) & (f_bins <= x_range[1])
        min_y = min(min_y, np.min(spectrum[within_range]))
        max_y = max(max_y, np.max(spectrum[within_range]))

    ax.set_xscale('symlog')
    # ax.set_xscale('log')
    ax.set_xlim([x_range[0], x_range[1]])
    ax.set_ylim([min_y, max_y])  # Set Y-axis limits based on the visible range

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Hex Hovering Spectral Plot')

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))

    fund_freq = 104
    ax.axvline(x=106, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'Fundamental: {106}')
    ax.axvline(x=209, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'1st Harmonic: {209}')
    ax.axvline(x=317, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'2nd Harmonic: {317}')
    ax.axvline(x=421, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'3rd Harmonic: {421}')
    ax.axvline(x=518, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'4th Harmonic: {518}')

    ax.grid(True, which='both')
    ax.legend(loc='upper right')

    fig.tight_layout(pad=1)
    plt.show()

# Function to see difference between hover spectrum and hover+vehicle spectrum
def hex_hover_vs_vehicle():
    hex_audio_2_filepath = '/Reference_Files/hex_hover_2.wav'
    hex_hover_2_audio = Audio_Abstract(filepath=hex_audio_2_filepath)
    hex_vehicle_audio_filepath = '/Reference_Files/hex_hover_vehicle.wav'
    hex_hover_vehicle_audio = Audio_Abstract(filepath=hex_vehicle_audio_filepath)

    frequency_range = (140, 3000)

    # Drone
    average_spectrum_hover, frequency_bins_d = process.average_spectrum(hex_hover_2_audio, frequency_range=frequency_range)
    # Drone + Vehicle
    average_spectrum_hover_vehicle, frequency_bins_dv = process.average_spectrum(hex_hover_vehicle_audio, frequency_range=frequency_range)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frequency_bins_d[:len(average_spectrum_hover)], average_spectrum_hover, color='b', alpha=0.5, label='Drone')
    ax.plot(frequency_bins_dv[:len(average_spectrum_hover_vehicle)], average_spectrum_hover_vehicle, color='g', alpha=0.5, label='Drone+Vehicle')
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Spectral Plot')
    ax.grid(True)
    fig.tight_layout(pad=1)
    plt.legend()
    plt.show()

# Function to see average hover vs hover+vehicle
def hex_hover_average_vs_vehicle():
    hex_vehicle_audio_filepath = '/Reference_Files/hex_hover_vehicle.wav'
    hex_hover_vehicle_audio = Audio_Abstract(filepath=hex_vehicle_audio_filepath)
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    frequency_range = (140, 3000)

    # Drone + Vehicle
    average_spectrum_hover_vehicle, frequency_bins_dv = process.average_spectrum(hex_hover_vehicle_audio, frequency_range=frequency_range)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(frequency_bins_dv[frequency_range[0]:len(average_spectrum_hover_vehicle)+frequency_range[0]], average_spectrum_hover_vehicle, color='g', alpha=0.99, label='Drone+Vehicle')

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, frequency_range=frequency_range)
        ax.plot(f_bins[frequency_range[0]:len(spectrum)+frequency_range[0]], spectrum, alpha=0.25)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Spectral Plot')
    ax.grid(True)
    fig.tight_layout(pad=1)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    hex_hover_average_spectrum()
    # hex_hover_vs_vehicle()
    # hex_hover_average_vs_vehicle()

    # base = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover'
    # # filepath = f'{base}/hex_hover_8_thin.wav'
    # filepath = f'{base}/Hex_8_Hover_3_d.wav'
    # audio = Audio_Abstract(filepath=filepath)
    # audio = process.normalize(audio)
    # spectrum, f_bins = process.average_spectrum(audio, norm=True, display=True)
    #
    #
    # harmonics = [(95, 117), (190, 224), (294, 336), (395, 445), (495, 540)]
    # average_psd_stats = hex_hover_stats(harmonics=harmonics)
    # # average_psd_stats = {1: {'power_sum': 647.5453383110874, 'count': 6138, 'average_psd': 0.10549777424423061},
    # #                      2: {'power_sum': 317.71822003686356, 'count': 9472, 'average_psd': 0.03354288640591887},
    # #                      3: {'power_sum': 353.6189515295496, 'count': 11694, 'average_psd': 0.030239349369723754},
    # #                      4: {'power_sum': 281.52192977653533, 'count': 13918, 'average_psd': 0.02022718276882708},
    # #                      5: {'power_sum': 202.39008765777646, 'count': 12528, 'average_psd': 0.016155019768341033}}
    #
    #
    # clean_spectrum = spectral_subtraction(spectrum, f_bins, average_psd_stats, harmonics=harmonics, display=True)

