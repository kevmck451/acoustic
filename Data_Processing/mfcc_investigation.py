from audio_abstract import Audio_Abstract
from Random.image_seq import generate_gif
import audio_filepaths as af

import process

from pathlib import Path



# Create MFCC Graphs
directory = Path(f'{af.basepath}/Synthetic/diesel_amb_mix_1')
for file in directory.iterdir():
    if 'wav' in str(file):
        audio = Audio_Abstract(filepath=file, num_channels=1)
        audio.mfccs = process.mfcc(audio, feature_params={'n_coeffs': 12}, display=True, save=True)


directory = Path(f'{af.basepath}/Synthetic/diesel_hex_mix_1')
for file in directory.iterdir():
    if 'wav' in str(file):
        audio = Audio_Abstract(filepath=file, num_channels=1)
        audio.mfccs = process.mfcc(audio, feature_params={'n_coeffs': 12}, display=True, save=True)


# Generate GIFs
base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
input_folder = f'{base_dir}/MFCC Graphs/combo levels/diesel hex'
output_folder = f'{base_dir}/MFCC Graphs/combo levels'
frame_duration = 0.05

name = 'amb_diesel_12.gif'
generate_gif(input_folder, output_folder, name, frame_duration)

name = 'hex_diesel_12.gif'
generate_gif(input_folder, output_folder, name, frame_duration)