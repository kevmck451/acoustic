

from Acoustic.process import average_spectrum
from Filters.audio import Audio


import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np



filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Angel/dataset'

freq_range = (400, 2500)




dataset_list = list()

for filepath in Path(filepath).glob('*.wav'):

    audio = Audio(filepath=filepath, num_channels=1)
    dataset_list.append(audio)

mean_empty_spectrum = list()
freq_list = list()

for audio in dataset_list:
    pass_num = audio.name.split('_')[1]
    target_type = audio.name.split('_')[0]

    if target_type == 'empty':
        spec, freq = average_spectrum(audio, frequency_range=freq_range, norm=True)
        freq_list = freq
        mean_empty_spectrum.append(spec)


mean_empty_spectrum = np.mean(mean_empty_spectrum, axis=0)

# plt.plot(freq_list, mean_empty_spectrum)

for audio in dataset_list:
    pass_num = audio.name.split('_')[1]
    target_type = audio.name.split('_')[0]

    if target_type != 'empty':
        spec, freq = average_spectrum(audio, frequency_range=freq_range, norm=True)
        spec = spec - mean_empty_spectrum
        spec[spec < 0] = 0
        plt.figure(figsize=(25, 10))
        plt.grid()
        plt.plot(freq_list, spec, label=audio.name)
        plt.legend()
        plt.title('Angel Boom Mount Analysis: Mean Subtracted')
        plt.tight_layout(pad=1)
        plt.show()


