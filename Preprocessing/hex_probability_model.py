

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af


import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from pathlib import Path
import numpy as np
from scipy.stats import skew
from scipy.stats import norm
import matplotlib.ticker as ticker


def hex_hover_std_skew(**kwargs):
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    harmonics = kwargs.get('harmonics', [(95, 117), (190, 224), (294, 336), (395, 445), (495, 540)])

    # Initialize dictionaries to hold std, skew values, mean frequency, and mean magnitude for each harmonic
    std_values = {i: [] for i in range(1, len(harmonics) + 1)}
    skew_values = {i: [] for i in range(1, len(harmonics) + 1)}
    mean_freq_values = {i: [] for i in range(1, len(harmonics) + 1)}
    mean_magnitude_values = {i: [] for i in range(1, len(harmonics) + 1)}

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, norm=True)

        # Calculate the std, skew, mean frequency, and mean magnitude within each harmonic range
        for i, (left_freq, right_freq) in enumerate(harmonics, start=1):
            indices = np.where((f_bins >= left_freq) & (f_bins <= right_freq))[0]
            if indices.size > 0:
                harmonic_freqs = f_bins[indices]
                harmonic_magnitudes = spectrum[indices]

                mean_frequency = np.average(harmonic_freqs, weights=harmonic_magnitudes)
                mean_freq_values[i].append(mean_frequency)

                mean_magnitude = np.mean(harmonic_magnitudes)
                mean_magnitude_values[i].append(mean_magnitude)

                freq_deviations = harmonic_freqs - mean_frequency
                std_deviation = np.std(freq_deviations)
                skewness = skew(freq_deviations)

                std_values[i].append(std_deviation)
                skew_values[i].append(skewness)
            else:
                print(f"No data in range {left_freq}-{right_freq} Hz for file {filepath}")

    # Calculate the mean std, skew, and mean frequency for each harmonic
    std_means = {i: np.round(np.mean(vals),3) for i, vals in std_values.items()}
    skew_means = {i: np.round(np.mean(vals),3) for i, vals in skew_values.items()}
    mean_freq_means = {i: np.round(np.mean(vals),3) for i, vals in mean_freq_values.items()}
    mean_magnitude_means = {i: np.round(np.mean(vals),3) for i, vals in mean_magnitude_values.items()}

    return mean_freq_means, std_means, skew_means, mean_magnitude_means

def real_distribution():
    harmonics_data = [
        {'mean': 106.268, 'std': 6.38, 'magnitude': 0.107},
        {'mean': 208.61, 'std': 9.844, 'magnitude': 0.031},
        {'mean': 316.658, 'std': 12.153, 'magnitude': 0.028},
        {'mean': 420.665, 'std': 14.463, 'magnitude': 0.018},
        {'mean': 518.005, 'std': 13.019, 'magnitude': 0.014}
    ]

    # Frequency range for plotting
    freq_range = np.linspace(0, 1000, 6000)

    # Set up the matplotlib figure and axes for the subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Plot each Gaussian distribution for each harmonic in the first subplot
    for harmonic in harmonics_data:
        distribution = norm(loc=harmonic['mean'], scale=harmonic['std'])
        ax[0].plot(freq_range, harmonic['magnitude'] * distribution.pdf(freq_range),
                    label=f"Harmonic at {harmonic['mean']} Hz")

    # Set titles and labels for the first subplot
    x_range = [80, 1000]
    ax[0].set_xlim([x_range[0], x_range[1]])
    ax[0].set_xscale('symlog')
    ax[0].set_title('Individual Harmonic Distributions')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax[0].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    ax[0].grid(True, which='both')

    # Combine the Gaussians for the complete model for the second subplot
    total_spectrum = np.sum(
        [harmonic['magnitude'] * norm(loc=harmonic['mean'], scale=harmonic['std']).pdf(freq_range) for harmonic in
         harmonics_data], axis=0)

    # Plot the combined model in the second subplot
    ax[1].plot(freq_range, total_spectrum, color='black', linestyle='--')
    x_range = [80, 1000]
    ax[1].set_xlim([x_range[0], x_range[1]])
    ax[1].set_xscale('symlog')
    ax[1].set_title('Combined Drone Ego Noise Frequency Spectrum Model')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend(['Combined Spectrum'])
    ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[1].xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax[1].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax[1].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    ax[1].grid(True, which='both')

    # Show the plots
    plt.tight_layout()
    plt.show()


def hex_hover_overlay():
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    harmonics_data = [
        {'mean': 106.268, 'std': 6.38, 'magnitude': 0.8*14},
        {'mean': 208.61, 'std': 9.844, 'magnitude': 0.3*20},
        {'mean': 316.658, 'std': 12.153, 'magnitude': 0.273*20},
        {'mean': 420.665, 'std': 14.463, 'magnitude': 0.206*20},
        {'mean': 518.005, 'std': 13.019, 'magnitude': 0.18*20}
    ]

    # Frequency range for plotting
    freq_range = np.linspace(0, 1000, 10000)
    x_range = [80, 1000]
    min_y, max_y = float('inf'), float('-inf')
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.axvline(x=106, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'Fundamental: {106}')
    ax.axvline(x=209, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'1st Harmonic: {209}')
    ax.axvline(x=317, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'2nd Harmonic: {317}')
    ax.axvline(x=421, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'3rd Harmonic: {421}')
    ax.axvline(x=518, color='blue', linestyle='-', linewidth=1, alpha=0.4, label=f'4th Harmonic: {518}')

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, norm=True)

        ax.plot(f_bins, spectrum, alpha=0.25)

        # Calculate Stats for Viewing
        within_range = (f_bins >= x_range[0]) & (f_bins <= x_range[1])
        min_y = min(min_y, np.min(spectrum[within_range]))
        max_y = max(max_y, np.max(spectrum[within_range]))

    # Combine the Gaussians for the complete model for the second subplot
    total_spectrum = np.sum(
        [harmonic['magnitude'] * norm(loc=harmonic['mean'], scale=harmonic['std']).pdf(freq_range) for harmonic in
         harmonics_data], axis=0)

    # Plot the combined model in the second subplot
    ax.plot(freq_range, total_spectrum, color='black', linestyle='--')

    ax.set_xscale('symlog')
    # ax.set_xscale('log')
    ax.set_xlim([x_range[0], x_range[1]])
    ax.set_ylim([min_y, max_y])  # Set Y-axis limits based on the visible range

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Real Data vs Model')

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))



    ax.grid(True, which='both')
    ax.legend(loc='upper right')

    fig.tight_layout(pad=1)
    plt.show()


if __name__ == '__main__':

    # ideal_distribution()


    # mean_means, std_means, skew_means, mag_means = hex_hover_std_skew()
    # print("Mean:", mean_means)
    # print("Std:", std_means)
    # print("Skewness :", skew_means)
    # print("Mags :", mag_means)

    # real_distribution()

    hex_hover_overlay()