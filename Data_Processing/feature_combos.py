
from audio_abstract import Audio_Abstract
import audio_filepaths as af
import process


import matplotlib.pyplot as plt
import numpy as np




# filepath = af.hex_hover_combo_thin
# filepath = af.hex_hover_combo_thick
# filepath = af.hex_hover_10m_1k_static1
# filepath = af.hex_hover_10m_static2
# filepath = af.angel_ff_1
# filepath = af.amb_orlando_1
# filepath = af.diesel_bulldozer_1_1

filepath = af.hex_diesel_100
# filepath = af.hex_diesel_40
# filepath = af.hex_diesel_0

audio = Audio_Abstract(filepath=filepath, num_channels=1)

# audio.mfccs = mfcc(audio, feature_params={'n_coeffs':12}, display=True)
# audio.mfccs = process.mfcc(audio, feature_params={'n_coeffs': 12}, display=False)
# print(audio.mfccs.shape)
#
# audio.av_spec, audio.av_spec_fb = process.average_spectrum(audio, display=True)
# print(audio.av_spec.shape)
# print(audio.av_spec_fb.shape)
#
# # check for duplicates
# if len(audio.av_spec_fb) == len(set(audio.av_spec_fb)):
#     print("All elements are unique.")
# else:
#     print("There are duplicates in the list.")

audio.spectrogram, audio.spec_freqs, audio.spec_times = process.spectrogram_2(audio, bandwidth=(0, 3000))
print(f'Spec Shape: {audio.spectrogram.shape}')
print(f'Freq Shape: {audio.spec_freqs.shape}')
print(f'Time Shape: {audio.spec_times.shape}')





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