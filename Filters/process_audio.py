
from Filters.noise_reduction import noise_reduction_filter
from Filters.high_pass import high_pass_filter
from Filters.low_pass import low_pass_filter
from Filters.normalize import normalize
from Filters.save_to_wav import save_to_wav
from Filters.audio import Audio

from pathlib import Path
import numpy as np

if __name__ == '__main__':

    base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data'

    filepath = f'{base_path}/Angel Noise Red/flights/A6_Flight_3.wav'
    audio = Audio(filepath=filepath, num_channels=4)

    filepath_noise = f'{base_path}/Angel Noise Red/profiles/angel_noise.wav'
    angel_noise = Audio(filepath=filepath, num_channels=4)

    shape_og = audio.data.shape
    print(audio)

    # Noise Reduction
    audio.data = noise_reduction_filter(audio, angel_noise, std_threshold=2.5)

    # High Pass Filter
    bottom_cutoff_freq = 200
    audio.data = high_pass_filter(audio, bottom_cutoff_freq)

    # Low Pass Filter
    top_cutoff_freq = 3000
    audio.data = low_pass_filter(audio, top_cutoff_freq)

    # Noise Reduction
    # audio.data = noise_reduction_filter(audio)

    # Ensure the filtered data shape matches the original
    assert audio.data.shape == shape_og, "Filtered data shape does not match original data shape"

    # Snip Ends which are garbage from NR
    audio.data = audio.data[:, 50000:-50000]

    # Normalize
    audio.data = normalize(audio)

    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')

    # Create the new filename
    original_path = Path(filepath)
    new_filename = original_path.stem + "_pr3" + original_path.suffix
    filepath_save = f'{base_path}/Angel Noise Red/output'
    new_filepath = f'{filepath_save}/{new_filename}'

    # Save the filtered audio to the new file
    save_to_wav(audio.data, audio.sample_rate, audio.num_channels, new_filepath)



