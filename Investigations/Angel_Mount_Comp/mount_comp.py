


from Filters.noise_reduction import noise_reduction_filter
from Filters.audio import Audio

from Acoustic.process import average_spectrum

from copy import deepcopy
import numpy as np



def estimate_noise(audio_filepath):

    nr_value = 1.5

    audio = Audio(filepath=audio_filepath, num_channels=1)
    audio_nr = deepcopy(audio)

    audio_nr.data = noise_reduction_filter(audio, noise_audio_object=None, std_threshold=nr_value)

    # Ensure the filtered data shape matches the original
    assert audio_nr.data.shape == audio.data.shape, "Filtered data shape does not match original data shape"

    og_average_spectrum, og_frequency_bins = average_spectrum(audio, display=False, norm=False)
    noise_average_spectrum, noise_frequency_bins = average_spectrum(audio_nr, display=False, norm=False)

    assert np.array_equal(og_frequency_bins, noise_frequency_bins), "Arrays are not equal!"

    estimated_noise_spectrum = og_average_spectrum - noise_average_spectrum
    estimated_noise_spectrum = np.clip(estimated_noise_spectrum, 0, None)

    ratio = np.sum(estimated_noise_spectrum) / np.sum(og_average_spectrum)

    return ratio

def calculate_ratio():
    a7_filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/Angel_7_empty_ch4.wav'
    a10_filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/Angel_10_empty_ch4.wav'

    a7_ratio = estimate_noise(a7_filepath)
    a10_ratio = estimate_noise(a10_filepath)

    print(f'A7 Ratio: {a7_ratio}')
    print(f'A10 Ratio: {a10_ratio}')

if __name__ == '__main__':

    # calculate_ratio()

    # Ratios
    A7 = 0.9516352794462176
    A10 = 0.9242695062955635

    # Calculate the percentage difference
    percentage_difference = ((A7 - A10) / A7) * 100

    # Print the result
    print(f"A10 ratio is {percentage_difference:.2f}% less than A7 ratio.")

    # path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/angel_prop_iso'
    # name = f'{audio_nr.name}_NR_{nr_value}_sfalse.wav'
    #
    # audio_nr.export(filepath=path, name=name)