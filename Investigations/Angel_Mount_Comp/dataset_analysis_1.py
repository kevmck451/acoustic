
from Acoustic.process import average_spectrum
from Filters.audio import Audio


import matplotlib.pyplot as plt
from pathlib import Path



filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Angel/dataset'

freq_range = (400, 2500)

plt.figure(figsize=(25,10))
plt.grid()

for filepath in Path(filepath).glob('*.wav'):

    audio = Audio(filepath=filepath, num_channels=1)
    pass_num = audio.name.split('_')[1]
    target_type = audio.name.split('_')[0]
    # print(audio.name)

    # if target_type != 'empty' and target_type != '500hz' and target_type != '1khz' and target_type != 'tonestack':
    if target_type == 'semi':
        spec, freq = average_spectrum(audio, frequency_range=freq_range, norm=True)
        plt.plot(freq, spec, label=audio.name)


plt.title('Angel Boom Mount Analysis')
plt.legend()
plt.tight_layout(pad=1)
plt.show()