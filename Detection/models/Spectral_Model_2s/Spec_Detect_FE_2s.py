

from Acoustic.audio import Audio

from copy import deepcopy
from pathlib import Path
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


# Load Data from a Dataset with Labels and Extract Features
def load_audio_data(path, duration=2):
    print('Loading Dataset')
    X = []
    y = []
    audio_ob_list = []

    sample_rate = 48000
    num_samples = sample_rate * duration

    for file in Path(path).rglob('*.wav'):
        audio = Audio_Spectral_Model(file)
        audio_copy = deepcopy(audio)
        start = 0
        end = num_samples
        total_samples = len(audio.data)

        # If the audio file is too short, pad it with zeroes
        if total_samples < num_samples:
            audio.data = np.pad(audio.data, (0, num_samples - len(audio.data)))
        # If the audio file is too long, shorten it
        elif total_samples > num_samples:
            while end <= total_samples:
                audio_copy.data = audio.data[start:end]
                audio_ob_list.append(audio_copy)
                start, end = (start + num_samples), (end + num_samples)
                try:
                    label = int(file.parent.stem)
                    y.append(label)  # Add Label (folder name)
                except:
                    continue

    print('Extracting Features')
    for audio in audio_ob_list:
        # Feature Extraction
        feature = audio.spectrogram()

        # print(feature.shape)
        # print(feature.dtype)

        X.append(feature)  # Add Feature

    X = np.array(X)
    X = X[..., np.newaxis]

    return X, np.array(y)