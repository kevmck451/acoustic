# File for starting point to detect targets from dataset


from Detection.models.dataset_info import *
from Acoustic.audio_multich import Audio_MC
from Acoustic.audio import Audio

from keras.models import load_model
import matplotlib.pyplot as plt
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

def full_flight_detection(directory):

    # LOAD DATA ------------------------------------------------------------------------
    print('Loading Dataset')

    sample_rate = 48000
    duration = 2
    num_samples = sample_rate * duration
    channel_list = []
    for file in Path(directory).rglob('*.wav'):
        audio_mch = Audio_MC(file)
        audio = Audio(file)
        audio_a = deepcopy(audio)
        audio_a.data = audio_mch.data[0]
        audio_b = deepcopy(audio)
        audio_b.data = audio_mch.data[1]
        audio_c = deepcopy(audio)
        audio_c.data = audio_mch.data[2]
        audio_d = deepcopy(audio)
        audio_d.data = audio_mch.data[3]
        channel_list = [audio_a, audio_b, audio_c, audio_d]


    audio_ob_list = []
    for audio in channel_list:

        start = 0
        end = num_samples
        total_samples = len(audio.data)

        # If the audio file is too short, pad it with zeroes
        if total_samples < num_samples:
            audio.data = np.pad(audio.data, (0, num_samples - len(audio.data)))
        # If the audio file is too long, shorten it
        elif total_samples > num_samples:
            audio_split_ob_list = []
            while end <= total_samples:
                audio_copy = deepcopy(audio)
                audio_copy.data = audio.data[start:end]
                audio_split_ob_list.append(audio_copy)
                start, end = (start + num_samples), (end + num_samples)
            audio_ob_list.append(audio_split_ob_list)


    # EXTRACT ------------------------------------------------------------------------
    print('Extracting Features')
    features_list = []
    for channel in audio_ob_list:
        feature_split_list = []
        for audio in channel:

            # Feature Extraction
            feature = audio.spectrogram()

            # print(feature.shape)
            # print(feature.dtype)

            feature_split_list.append(feature)  # Add Feature
        features_list.append(np.array(feature_split_list))

    for i in range(len(features_list)):
        features_list[i] = features_list[i][..., np.newaxis]

    # PREDICT ------------------------------------------------------------------------
    print('Making Predictions')
    model = load_model('models/Spectral_Model_2s/model_library/detect_spec_2_88_0.h5')


    predictions_list = []

    for channel in features_list:
        predictions = []
        for feature in channel:
            feature = np.expand_dims(feature, axis=0)
            y_new_pred = model.predict(feature)
            y_pred_class = int(y_new_pred[0][0] > 0.5)  # Convert to binary class prediction
            percent = np.round((y_new_pred[0][0] * 100), 2)
            predictions.append(percent)

        predictions_list.append(predictions)

    time = list(range(0, (len(predictions_list[0]) * 2), 2))
    # AVERAGE CHANNEL PREDICTIONS ------------------------------------------------------------------------
    print(time)
    print(predictions_list)

    import statistics

    averaged_predictions = [statistics.mean(values) for values in zip(*predictions_list)]

    print(averaged_predictions)

    fig, axs = plt.subplots(5, 1, figsize=(14, 6))
    plt.suptitle('Sound Source Detection')

    # Loop over your 4 lists
    for i in range(4):
        bar_colors = ['g' if value >= 50 else 'r' for value in predictions_list[i]]
        axs[i].bar(time, predictions_list[i], color=bar_colors)
        axs[i].set_title(f"Channel {i + 1}")
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Predictions')
        axs[i].axhline(50, c='black', linestyle='dotted')

    # Plot averaged_predictions
    bar_colors = ['g' if value >= 50 else 'r' for value in averaged_predictions]
    axs[4].bar(time, averaged_predictions, color=bar_colors)
    axs[4].set_title(f"Averaged Predictions")
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Predictions')
    axs[4].axhline(50, c='black', linestyle='dotted')

    # Ensure the subplots do not overlap
    plt.tight_layout(pad=1)
    plt.show()



if __name__ == '__main__':

    full_flight_detection(directory_mission_5)

