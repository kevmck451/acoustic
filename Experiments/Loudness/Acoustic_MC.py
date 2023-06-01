# Audio Class for Analyzing WAV Files
# Kevin McKenzie 2023


import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import librosa




from prettytable import PrettyTable

from tabulate import tabulate
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.fft import fft
import soundfile as sf
import wave
import os



class Audio_MC:
    def __init__(self, filepath, stats=False):

        # File Name Processing
        self.filepath = filepath
        n = filepath.split('/')
        self.filename = n[-1]
        self.directory = '/'.join(n[:-1]) + '/'

        # Open the wave file in read mode
        self.data, self.sample_rate = sf.read(filepath, dtype='int16')

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
        stat_names = ['Channel', 'Max Value', 'Min Value', 'Mean', 'Std Dev', 'RMS', 'Dynamic Range', 'Average']
        channel_stats = []

        for i in range(self.channels):
            channel_data = self.data[i, :]
            max_value = np.max(channel_data).round(3)
            min_value = np.min(channel_data).round(3)
            mean = np.mean(channel_data).round(3)
            std_dev = np.std(channel_data).round(3)
            rms = np.sqrt(np.mean(np.square(channel_data))).round(3)
            dynamic_range = max_value - min_value  # Calculate dynamic range
            channel_stats.append({
                stat_names[0]: i + 1,
                stat_names[1]: max_value,
                stat_names[2]: min_value,
                stat_names[3]: mean,
                stat_names[4]: std_dev,
                stat_names[5]: rms,
                stat_names[6]: dynamic_range  # Include dynamic range in the stats dictionary
            })

        # Compute average values for all channels
        avg_max_value = np.mean([stat[stat_names[1]] for stat in channel_stats]).round(3)
        avg_min_value = np.mean([stat[stat_names[2]] for stat in channel_stats]).round(3)
        avg_mean = np.mean([stat[stat_names[3]] for stat in channel_stats]).round(6)
        avg_std_dev = np.mean([stat[stat_names[4]] for stat in channel_stats]).round(3)
        avg_rms = np.mean([stat[stat_names[5]] for stat in channel_stats]).round(3)
        avg_dynamic_range = np.mean([stat[stat_names[6]] for stat in channel_stats]).round(
            3)  # Calculate average dynamic range

        # Add a row for average values
        channel_stats.append({
            stat_names[0]: stat_names[7],
            stat_names[1]: avg_max_value,
            stat_names[2]: avg_min_value,
            stat_names[3]: avg_mean,
            stat_names[4]: avg_std_dev,
            stat_names[5]: avg_rms,
            stat_names[6]: avg_dynamic_range  # Include average dynamic range
        })

        if display:
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

            plt.show()

        return channel_stats

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


class Compare:
    def __init__(self, directory):
        self.directory = directory
        self.sample_list = []
        self.channels = 4

        for filename in os.listdir(directory):
            # Check if the file is a file (not a subfolder)
            self.filepath = os.path.join(directory, filename)
            if os.path.isfile(self.filepath):
                if filename.endswith('.wav'):
                    self.sample_list.append(Audio_MC(self.filepath))
                    # print(self.filepath)

        # print(self.files_list)
        self.sample_list.sort(key=lambda x: x.filename)

    def RMS(self, title):

        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Transpose the RMS values
        rms_values = []

        for i in range(len(channel_labels)):
            rms_values.append([sample.stats()[i]['RMS'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, rms in enumerate(rms_values):
            plt.bar([p + barWidth * i for p in r1], rms, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('RMS', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} RMS Comparison')
        plt.legend()
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                rms = stats['RMS']
                print(f"Channel {channel}: RMS={rms}")
            print()

    def Peak(self, title):
        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Transpose the peak level values
        peak_values = []

        for i in range(len(channel_labels)):
            peak_values.append([sample.stats()[i]['Max Value'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, peak in enumerate(peak_values):
            plt.bar([p + barWidth * i for p in r1], peak, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('Peak Level', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} Peak Level Comparison')
        plt.legend()
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                peak = stats['Max Value']
                print(f"Channel {channel}: Peak Level={peak}")
            print()

    def Range(self, title):
        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Calculate the dynamic range values
        dynamic_range_values = []

        for i in range(len(channel_labels)):
            dynamic_range_values.append([sample.stats()[i]['Dynamic Range'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, dynamic_range in enumerate(dynamic_range_values):
            plt.bar([p + barWidth * i for p in r1], dynamic_range, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('Dynamic Range', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} Dynamic Range Comparison')
        plt.legend()
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                dynamic_range = stats['Dynamic Range']
                print(f"Channel {channel}: Dynamic Range={dynamic_range}")
            print()

    def Spectral(self, title):
        # Prepare data for plotting
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]

        # Define the desired frequency range
        frequency_range = (0, 2000)

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for i, sample in enumerate(self.sample_list):
            channel_spectrums = []
            for channel_data in sample.data:
                spectrum = np.fft.fft(channel_data)  # Apply FFT to the audio data
                magnitude = np.abs(spectrum)

                # Calculate frequency bins and positive frequency mask for each sample
                frequency_bins = np.fft.fftfreq(len(channel_data), d=1 / sample.sample_rate)
                positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])

                channel_spectrums.append(magnitude[positive_freq_mask][:len(frequency_bins)])

            # Average across all channels
            average_spectrum = np.mean(channel_spectrums, axis=0)

            ax.plot(frequency_bins[:len(average_spectrum)], average_spectrum, color=colors[i % len(colors)],
                    label=sample_names[i])

        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude', fontweight='bold')
        ax.set_title(f'{title} Spectral Analysis')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()