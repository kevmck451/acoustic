

import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import wave


class Audio_Abstract:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('filepath', None))
        self.sample_rate = kwargs.get('sample_rate', 48000)
        self.num_channels = kwargs.get('num_channels', 1)
        self.sample_length = kwargs.get('sample_length', None)
        self.data = kwargs.get('data', None)
        self.num_samples = kwargs.get('num_samples', None)
        self.name = kwargs.get('name', self.path.stem)

        # If given a filepath
        if self.path is not None:
            # with wave.open(str(self.path), 'rb') as wav_file:
            #     self.num_channels = wav_file.getnchannels()
            info = sf.info(self.path)
            self.num_channels = info.channels
            self.load_data(self.path)

    def __str__(self):
        return f'---------Audio Object---------\n' \
               f'sample_rate: {self.sample_rate} Hz\n' \
               f'num_channels: {self.num_channels}\n' \
               f'sample_length: {self.sample_length} s\n' \
               f'num_samples: {self.num_samples}\n' \
               f'data type: {self.data.dtype}\n' \
               f'data shape: {self.data.shape}\n' \
               f'data: {self.data}'

    # Function that loads data from filepath
    def load_data(self, filepath):
        if self.num_channels > 1:
            self.data, samplerate = sf.read(str(filepath), dtype='float32')
            if samplerate != self.sample_rate:
                self.data = librosa.resample(y=self.data, orig_sr=samplerate, target_sr=self.sample_rate)

            try:
                self.data = self.data.reshape(-1, self.num_channels)  # Reshape to match the number of channels
            except ValueError:
                print("The audio data cannot be reshaped to match the number of channels.")
                return

            # Convert the interleaved data to deinterleaved format
            self.data = np.transpose(self.data.copy())  # Rows are channels / columns are data
            self.sample_length = round((self.data.shape[1] / self.sample_rate), 2)
            self.num_samples = len(self.data[1])

        else:
            self.data, samplerate = sf.read(str(filepath), dtype='float32')
            if samplerate != self.sample_rate:
                for i, channel in enumerate(self.data):
                    self.data[i] = librosa.resample(channel, samplerate, self.sample_rate)
            self.sample_length = round((len(self.data) / self.sample_rate), 2)
            self.num_samples = len(self.data)

    # Function that returns stats from the audio file
    def stats(self):
        stat_names = ['Max', 'Min', 'Mean', 'RMS', 'Range']

        max_value = np.max(self.data).round(3)
        min_value = np.min(self.data).round(3)
        mean = np.mean(self.data).round(3)
        rms = np.sqrt(np.mean(np.square(self.data)))
        dynamic_range = (max_value - min_value).round(3)  # Calculate dynamic range
        channel_stats = {
            stat_names[0]: max_value,
            stat_names[1]: min_value,
            stat_names[2]: mean,
            stat_names[3]: rms,
            stat_names[4]: dynamic_range
            }

        return channel_stats

    # Function to export an object
    def export(self, **kwargs):
        filepath = kwargs.get('filepath', None)
        # Save/export the audio object
        if filepath is not None:
            if Path(filepath).suffix != '.wav':
                filepath = filepath + '.wav'
            sf.write(f'{filepath}', self.data, self.sample_rate)
        else: sf.write(f'{self.name}_export.wav', self.data, self.sample_rate)

    # Function to display the waveform of audio
    def waveform(self):
        # Calculate the time axis in seconds
        time_axis = np.arange(len(self.data[0])) / self.sample_rate

        # Create the figure and axis objects, with subplots equal to number of channels
        fig, axs = plt.subplots(self.num_channels, figsize=(14, (4+self.num_channels)))

        # If only one channel, make axs a list to handle indexing
        if self.num_channels == 1:
            axs = [axs]

        # Plot the audio data for each channel
        for i in range(self.num_channels):
            axs[i].plot(time_axis, self.data[i], linewidth=0.5)
            axs[i].set_ylabel('Amplitude')
            axs[i].set_ylim([-1, 1])  # set the y-axis limits to -1 and 1
            axs[i].axhline(y=0, color='black', linewidth=0.5, linestyle='--')  # add horizontal line at y=0
            axs[i].set_xlabel('Time (s)')
            axs[i].set_title(f'Waveform: {self.path.stem} - Channel {i + 1}')

        # Make the layout tight to avoid overlap of subplots
        fig.tight_layout(pad=1)
        plt.show()


if __name__ == '__main__':

    a = Audio_Abstract(stats=True)

    print('-' * 50)

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/mission 5/Hex_FullFlight_5.wav'
    b = Audio_Abstract(filepath=filepath, stats=True)

    print('-'*50)

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset 5/1/5_target_1_a.wav'
    c = Audio_Abstract(filepath=filepath, stats=True)





