# Audio Class for Analyzing WAV Files
# Kevin McKenzie 2022

from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.fft import fft
import soundfile as sf
import librosa.display
import numpy as np
import librosa
import wave
import os


class Audio:
    def __init__(self, filepath, stats):

        # File Name Processing
        self.filepath = filepath
        n = filepath.split('/')
        self.filename = n[-1]
        self.directory = '/'.join(n[:-1]) + '/'

        # Open the wave file in read mode
        self.data, self.sample_rate = sf.read(filepath, dtype='float64')

        # The data array contains the audio data with four channels
        # Each row represents a single sample across all four channels
        self.channels = 4
        self.data = self.data.reshape(-1, self.channels) # Reshape to 4 Channel
        self.data = self.data - np.mean(self.data) # Remove bias / center around 0
        self.sample_length = round((self.data.shape[0] / self.sample_rate), 2)

        # Convert the interleaved data to deinterleaved format
        self.data = self.data.transpose() # Rows are channels / columns are data

        if stats:
            print(f'Filepath: {self.filepath}')
            print(f'Directory: {self.directory}')
            print(f'Filename: {self.filename}')
            print(f'Sample Rate: {self.sample_rate} Hz')
            print(f'Sample Length: {self.sample_length} s')
            print(f'Data Shape: {self.data.shape}')
            # print(self.data)
            # print()
            # print(self.data[0])

    def export(self):
        pass

    def stats(self, display=False):

        # Compute various statistics for each channel
        channel_stats = []
        for i in range(self.channels):
            channel_data = self.data[i, :]
            max_value = np.max(channel_data).round(3)
            min_value = np.min(channel_data).round(3)
            mean = np.mean(channel_data).round(3)
            std_dev = np.std(channel_data).round(3)
            rms = np.sqrt(np.mean(np.square(channel_data))).round(3)
            channel_stats.append((f'Channel {i + 1}', max_value, min_value, mean, std_dev, rms))

        # Compute average values for all channels
        avg_max_value = np.mean([stat[1] for stat in channel_stats]).round(3)
        avg_min_value = np.mean([stat[2] for stat in channel_stats]).round(3)
        avg_mean = np.mean([stat[3] for stat in channel_stats]).round(6)
        avg_std_dev = np.mean([stat[4] for stat in channel_stats]).round(3)
        avg_rms = np.mean([stat[5] for stat in channel_stats]).round(3)

        # Add a row for average values
        channel_stats.append(('Average', avg_max_value, avg_min_value, avg_mean, avg_std_dev, avg_rms))

        # Create a table to display the statistics
        table = PrettyTable(['Channel', 'Max Value', 'Min Value', 'Mean', 'Std Dev', 'RMS'])
        for row in channel_stats:
            table.add_row(row)

        # Create a figure and add the table to it
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.axis('off')
        ax.table(cellText=[table.field_names] + table._rows, colWidths=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1], cellLoc='center',
                 loc='center')
        ax.set_title(f'Stats: {self.filename}')

        if display:
            plt.show()

        return channel_stats[0][1]

    def visualize(self, channel=1):
        # Calculate the time axis in seconds
        time_axis = np.arange(self.data.shape[1]) / self.sample_rate

        # Plot the specified channel of the audio data
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_axis, self.data[channel - 1, :], linewidth=0.5)
        ax.set_ylabel(f'Channel {channel}')
        ax.set_ylim([-1, 1])  # set the y-axis limits to -1 and 1
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')  # add horizontal line at y=0
        plt.xlabel('Time (s)')
        plt.title(f'Visualize: {self.filename}, Channel {channel}')
        plt.show()

    def visualize_4ch(self):
        # Calculate the time axis in seconds
        time_axis = np.arange(self.data.shape[1]) / self.sample_rate

        # Plot all four channels of the audio data
        fig, axs = plt.subplots(nrows=self.channels, sharex=True, sharey=True, figsize=(14, 8))
        plt.suptitle(f'Visualize: {self.filename}')
        for i in range(self.channels):
            axs[i].plot(time_axis, self.data[i, :], linewidth=0.5)
            axs[i].set_ylabel(f'Channel {i + 1}')
            axs[i].set_ylim([-1, 1])  # set the y-axis limits to -1 and 1
            axs[i].axhline(y=0, color='black', linewidth=0.5, linestyle='--')  # add horizontal line at y=0
        plt.xlabel('Time (s)')
        plt.tight_layout(pad=1)
        plt.show()

    def spectro(self, channel=1, log=False, freq=(20, 2000)):

        # Define the frequency axis for the spectrogram
        fft_size = 4096  # 1024
        freq_axis = np.fft.rfftfreq(fft_size, 1 / self.sample_rate)

        # Plot the spectrogram for the specified channel of the audio data
        fig, ax = plt.subplots(figsize=(10, 12))
        spec_data, freq_axis, time_axis, img = ax.specgram(self.data[channel - 1, :], Fs=self.sample_rate,
                                                           NFFT=fft_size,
                                                           noverlap=0, cmap='hot_r', vmin=-100, vmax=0)
        ax.set_ylabel(f'Freq (Hz)')
        if log:
            ax.set_yscale('log')
        ax.set_ylim([freq[0], freq[1]])  # set the y-axis limits to 0 and 20000 Hz
        plt.xlabel('Time (s)')
        plt.title(f'Spectrogram: {self.filename}, Channel {channel}')
        plt.colorbar(img)
        plt.show()

    def spectro_4ch(self, log=False, freq=(20, 2000)):

        # Define the frequency axis for the spectrogram
        fft_size = 4096 # 1024
        freq_axis = np.fft.rfftfreq(fft_size, 1 / self.sample_rate)

        # Plot the spectrogram for each channel of the audio data
        fig, axs = plt.subplots(nrows=self.channels, sharex=True, sharey=True, figsize=(14, 8))
        plt.suptitle(f'Spectrogram: {self.filename}')
        for i in range(self.data.shape[0]):
            spec_data, freq_axis, time_axis, img = axs[i].specgram(self.data[i, :], Fs=self.sample_rate, NFFT=fft_size,
                                                                   noverlap=0, cmap='hot_r', vmin=-100, vmax=0)
            axs[i].set_ylabel(f'CH{i + 1}: Freq (Hz)')
            if log:
                axs[i].set_yscale('log')
            axs[i].set_ylim([freq[0], freq[1]])  # set the y-axis limits to 0 and 20000 Hz


        plt.xlabel('Time (s)')
        plt.tight_layout(pad=1)
        # Add a colorbar
        cbar = plt.colorbar(img, ax=axs)

        plt.show()


class Acoustic:
    def __init__(self):
        pass
