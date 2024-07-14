


from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process

import matplotlib.pyplot as plt

base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Angel/2 Test 2'
flight_names = ['Angel_6', 'Angel_7']
angel_6_filepath = f'{base_path}/{flight_names[0]}_flight.wav'
angel_7_filepath = f'{base_path}/{flight_names[1]}_flight.wav'

angel_6_audio = Audio_Abstract(filepath=angel_6_filepath, num_channels=4)
angel_7_audio = Audio_Abstract(filepath=angel_7_filepath, num_channels=4)

angel_6_audio_ch_list = process.channel_to_objects(angel_6_audio)
angel_7_audio_ch_list = process.channel_to_objects(angel_7_audio)

feature_params = {'bandwidth': (70, 2500), 'nperseg': 8192*(2**2)}
for channel_6, channel_7 in zip(angel_6_audio_ch_list, angel_7_audio_ch_list):
    channel_6.spectrograms, channel_6.frequencies, channel_6.times = process.spectrogram_2(channel_6, feature_params=feature_params, details=True)
    channel_7.spectrograms, channel_7.frequencies, channel_7.times = process.spectrogram_2(channel_7, feature_params=feature_params, details=True)


angel_6_spec = 0
angel_7_spec = 1

fig, axs = plt.subplots(2, 1, figsize=(18, 9))
plt.suptitle('Bullet Mic Mount Wind Muff Comparison')

y_ticks = [feature_params.get('bandwidth')[0], 125, 250, 500, 1000, 1500, 2000, feature_params.get('bandwidth')[1]]
y_labels = [str(feature_params.get('bandwidth')[0]), '125', '250', '500', '1k', '1.5k', '2k', '2.5k']

angel_6_channel = angel_7_channel = 1

# Angel 6 Spectrogram
axs[angel_6_spec].set_title(f'Angel 6 Ch{angel_6_channel}: Spectrogram')
im_6 = axs[angel_6_spec].pcolormesh(angel_6_audio_ch_list[angel_6_channel-1].times, angel_6_audio_ch_list[angel_6_channel-1].frequencies, angel_6_audio_ch_list[angel_6_channel-1].spectrograms, shading='gouraud', vmin=0, vmax=1)
axs[angel_6_spec].set_xlabel('Time (s)')
axs[angel_6_spec].set_ylabel(f"Frequency: {feature_params.get('bandwidth')[0]} - {feature_params.get('bandwidth')[1]}")
fig.colorbar(im_6, ax=axs[angel_6_spec], label='Intensity', extend='both')
axs[angel_6_spec].set_yscale('log')
axs[angel_6_spec].set_yticks(y_ticks)
axs[angel_6_spec].set_yticklabels(y_labels)
axs[angel_6_spec].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.25, color='black')
axs[angel_6_spec].minorticks_on()

# Angel 7 Spectrogram
axs[angel_7_spec].set_title(f'Angel 6 Ch{angel_7_channel}: Spectrogram')
im_7 = axs[angel_7_spec].pcolormesh(angel_7_audio_ch_list[angel_7_channel-1].times, angel_7_audio_ch_list[angel_7_channel-1].frequencies, angel_7_audio_ch_list[angel_7_channel-1].spectrograms, shading='gouraud', vmin=0, vmax=1)
axs[angel_7_spec].set_xlabel('Time (s)')
axs[angel_7_spec].set_ylabel(f"Frequency: {feature_params.get('bandwidth')[0]} - {feature_params.get('bandwidth')[1]}")
fig.colorbar(im_7, ax=axs[angel_7_spec], label='Intensity', extend='both')
axs[angel_7_spec].set_yscale('log')
axs[angel_7_spec].set_yticks(y_ticks)
axs[angel_7_spec].set_yticklabels(y_labels)
axs[angel_7_spec].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.25, color='black')
axs[angel_7_spec].minorticks_on()

plt.tight_layout(pad=1)
plt.show()











