
from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import process
from Acoustic import utils
import audio_filepaths as af

import numpy as np



if __name__ == '__main__':
    # filepath = af.full_flight_dynamic_1b
    filepath = af.full_flight_static_3_test_2
    audio = Audio_Abstract(filepath=filepath, num_channels=1)

    # audio.av_spec, audio.av_spec_fb = process.average_spectrum(audio, display=True)
    # print(audio.av_spec.shape)

    audio.spectrogram, audio.spec_freqs, audio.spec_times = process.spectrogram_2(audio, stats=False,
                                                                                  feature_params={'bandwidth':(70, 3000)},
                                                                                  display=True, details=True, norm=True)
    # print(f'Max: {np.max(audio.spectrogram)}\nMin: {np.min(audio.spectrogram)}\nMean: {np.mean(audio.spectrogram)}')


