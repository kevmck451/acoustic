
from audio_abstract import Audio_Abstract
import audio_filepaths as af
import process


import matplotlib.pyplot as plt





# filepath = af.hex_hover_combo_thin
# filepath = af.hex_hover_combo_thick
# filepath = af.hex_hover_10m_1k_static1
# filepath = af.hex_hover_10m_static2
# filepath = af.angel_ff_1
# filepath = af.amb_orlando_1
# filepath = af.diesel_bulldozer_1_1

filepath = af.hex_diesel_99
# filepath = af.hex_diesel_59
# filepath = af.hex_diesel_1

audio = Audio_Abstract(filepath=filepath, num_channels=1)

# audio.mfccs = mfcc(audio, feature_params={'n_coeffs':12}, display=True)
audio.mfccs = process.mfcc(audio, feature_params={'n_coeffs': 12}, display=False)
print(audio.mfccs.shape)
plt.imshow(audio.mfccs)
plt.tight_layout(pad=1)
plt.show()

audio.av_spec, audio.av_spec_fb = process.average_spectrum(audio, display=True)
print(audio.av_spec.shape)

# audio.spectrogram = spectrogram(audio, stats=False, feature_params={'bandwidth': (0, 20000)}, display=True)
# audio.spectrogram, audio.spec_freqs, audio.spec_times = spectrogram(audio, stats=False, feature_params={'bandwidth':(0, 24000)}, display=False, details=True, norm=True)
# print(f'Max: {np.max(audio.spectrogram)}\nMin: {np.min(audio.spectrogram)}\nMean: {np.mean(audio.spectrogram)}')
#
# fig, ax = plt.subplots()
# for i in range(0, len(audio.spec_times)):
#     if i%20 == 0:
#         ax.plot(audio.spec_freqs, audio.spectrogram[:, i])
#         ax.set_xlabel('Frequency (Hz)', fontweight='bold')
#         ax.set_ylabel('Magnitude', fontweight='bold')
#         ax.set_title(f'Spectral Plot: {audio.name}')
#         ax.grid(True)
#         fig.tight_layout(pad=1)
# plt.show()

# audio_list, _ = generate_windowed_chunks(audio, window_size=0.1)
#
# for audio in audio_list:
#     audio.av_spec = average_spectrum(audio, display=False)
# plt.show()