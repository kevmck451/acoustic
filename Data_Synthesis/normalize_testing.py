

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process

import numpy as np

synthetic_directory = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Synthetic'


noise_floor_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Isolated Samples/Ambient/residential_amb_2-1.wav'
noise_floor = Audio_Abstract(filepath=noise_floor_path)
noise_floor.export(filepath = synthetic_directory, name = f'{noise_floor.name}')
# noise_floor.waveform()

noise_floor = process.normalize(noise_floor, percentage=100)
# noise_floor.waveform()
noise_floor.export(filepath = synthetic_directory, name = f'{noise_floor.name}_norm')


noise_floor_path = f'{synthetic_directory}/residential_amb_2-1_norm.wav'
noise_floor = Audio_Abstract(filepath=noise_floor_path)
# noise_floor.waveform()

noise_floor = process.normalize(noise_floor, percentage=15)
# noise_floor.waveform()
noise_floor.export(filepath = synthetic_directory, name = f'{noise_floor.name}_renorm')