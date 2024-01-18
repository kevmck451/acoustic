from audio_abstract import Audio_Abstract
from Investigations.Features.image_seq import generate_gif
import audio_filepaths as af

import process

from pathlib import Path


base_dir = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Analysis'
output_folder = f'{base_dir}/ZCR Graphs/combo levels'
frame_duration = 0.05

# Create MFCC Graphs
directory = Path(f'{af.basepath}/Synthetic/diesel_amb_mix_3')
save_path = f'{base_dir}/ZCR Graphs/combo levels/diesel amb'
Path(save_path).mkdir(exist_ok=True, parents=True)
for file in directory.iterdir():
    if 'wav' in str(file):
        audio = Audio_Abstract(filepath=file, num_channels=1)
        audio.zcr = process.zcr(audio, display=True, save=True, save_path=save_path)

# Generate GIFs
input_folder = save_path
name = 'amb_diesel_zcr.gif'
generate_gif(input_folder, output_folder, name, frame_duration)


# ---------------------------------------------------------------------------


# Create MFCC Graphs
directory = Path(f'{af.basepath}/Synthetic/diesel_hex_mix_3')
save_path = f'{base_dir}/ZCR Graphs/combo levels/diesel hex'
Path(save_path).mkdir(exist_ok=True, parents=True)
for file in directory.iterdir():
    if 'wav' in str(file):
        audio = Audio_Abstract(filepath=file, num_channels=1)
        audio.zcr = process.zcr(audio, display=True, save=True, save_path=save_path)

# Generate GIFs
input_folder = save_path
name = 'hex_diesel_zcr.gif'
generate_gif(input_folder, output_folder, name, frame_duration)