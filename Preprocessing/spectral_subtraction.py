from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process

import matplotlib.pyplot as plt
from pathlib import Path


# Function to see average spectrum for hex hovering
def hex_hover_average_spectrum():
    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    fig, ax = plt.subplots(figsize=(14, 4))

    for filepath in directory.rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, frequency_range=(140, 3000))
        ax.plot(f_bins[:len(spectrum)], spectrum, alpha=0.25)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Spectral Plot')
    ax.grid(True)
    fig.tight_layout(pad=1)
    plt.show()

# Function to see difference between hover spectrum and hover+vehicle spectrum
def hex_hover_vs_vehicle():
    hex_audio_2_filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Reference_Files/hex_hover_2.wav'
    hex_hover_2_audio = Audio_Abstract(filepath=hex_audio_2_filepath)
    hex_vehicle_audio_filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Reference_Files/hex_hover_vehicle.wav'
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
    hex_vehicle_audio_filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Reference_Files/hex_hover_vehicle.wav'
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

    # hex_hover_average_spectrum()
    # hex_hover_vs_vehicle()
    hex_hover_average_vs_vehicle()