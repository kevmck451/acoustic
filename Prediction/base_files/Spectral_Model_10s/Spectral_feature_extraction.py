

from Acoustic.audio import Audio

import numpy as np
import librosa
import process


class Audio_Spectral_Model(Audio):
    def __init__(self, path):
        super().__init__(path)

    # Function to calculate spectrogram of audio
    def spectrogram(self, range=(80, 2000), stats=False):
        # Do not change settings - ML Model depends on it as currently set
        window_size = 32768
        hop_length = 512
        frequency_range = range

        Audio_Object = process.normalize(self)
        data = Audio_Object.data

        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = np.abs(librosa.stft(data, n_fft=window_size, hop_length=hop_length)) ** 2

        # Convert to decibels (log scale) for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Calculate frequency range and resolution
        nyquist_frequency = self.SAMPLE_RATE / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)
        frequency_range = np.arange(0, window_size // 2 + 1) * frequency_resolution
        self.freq_range_low = int(frequency_range[0])
        self.freq_range_high = int(frequency_range[-1])
        self.freq_resolution = round(frequency_resolution, 2)

        bottom_index = int(np.round(range[0] / frequency_resolution))
        top_index = int(np.round(range[1] / frequency_resolution))

        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range: ({range[0]},{range[1]}) Hz')
            print(f'Freq Resolution: {self.freq_resolution} Hz')

        return spectrogram_db[bottom_index:top_index]


def extract_features(path, duration):
    num_samples = 48000 * duration
    # Load audio file with fixed sample rate
    audio = Audio_Spectral_Model(path)

    # If the audio file is too short, pad it with zeroes
    if len(audio.data) < num_samples:
        audio.data = np.pad(audio.data, (0, num_samples - len(audio.data)))
    # If the audio file is too long, shorten it
    elif len(audio.data) > num_samples:
        audio.data = audio.data[:num_samples]

    # Feature Extraction
    spectro = audio.spectrogram()

    return spectro