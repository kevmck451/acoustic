# Audio Class for Analyzing WAV Files
# Kevin McKenzie 2023

from pathlib import Path
import numpy as np
import librosa
import Sample_Library
from Utils import CSVFile
import Process
import soundfile as sf

class Audio:

    def __init__(self, filepath, SR=Sample_Library.SAMPLE_LIBRARY_SAMPLE_RATE, stats=False):
        self.filepath = Path(filepath)
        self.SAMPLE_RATE = SR
        self.CHANNEL_NUMBER = 1

        self.data, _ = librosa.load(filepath, sr=self.SAMPLE_RATE)
        self.sample_length = round((self.data.shape[0] / self.SAMPLE_RATE), 2)

        # Initial Conditions of Audio File
        CSV = CSVFile(Sample_Library.SAMPLE_LIBRARY_LIST)
        self.location = CSV.get_value(self.filepath.stem, 'Location')
        self.date = CSV.get_value(self.filepath.stem, 'Date')
        self.time = CSV.get_value(self.filepath.stem, 'Time')
        self.vehicle = CSV.get_value(self.filepath.stem, 'Vehicle')
        self.recorder = CSV.get_value(self.filepath.stem, 'Recorder')
        self.mount = CSV.get_value(self.filepath.stem, 'Mount')
        self.cover = CSV.get_value(self.filepath.stem, 'Cover')
        self.position = CSV.get_value(self.filepath.stem, 'Position')
        self.raw = CSV.get_value(self.filepath.stem, 'RAW')
        self.category = CSV.get_value(self.filepath.stem, 'Category')
        self.temp = CSV.get_value(self.filepath.stem, 'Temp')
        self.humidity = CSV.get_value(self.filepath.stem, 'Humidity')
        self.pressure = CSV.get_value(self.filepath.stem, 'Pressure')
        self.wind = CSV.get_value(self.filepath.stem, 'Wind')

        if stats:
            print(f'File Name: {self.filepath.name}')
            print(f'Data Shape: {self.data.shape}')
            print(f'Date Type: {self.data.dtype}')
            print(f'Sample Length: {self.sample_length} s')
            print(f'Sample Rate: {self.SAMPLE_RATE} Hz')
            print(f'Channel Number: {self.CHANNEL_NUMBER}')
            print(f'Location: {self.location}')
            print(f'Date: {self.date}')
            print(f'Time: {self.time}')
            print(f'Vehicle: {self.vehicle}')
            print(f'Recorder: {self.recorder}')
            print(f'Mount: {self.mount}')
            print(f'Cover: {self.cover}')
            print(f'Position: {self.position}')
            print(f'RAW: {self.raw}')
            print(f'Type: {self.category}')
            print(f'Temperature: {self.temp} F')
            print(f'Humidity: {self.humidity} %')
            print(f'Pressure: {self.pressure} hPa')
            print(f'Wind: {self.wind} m/s')

    # Function that returns stats from the audio file
    def stats(self):
        stat_names = ['Max', 'Min', 'Mean', 'RMS', 'Range']
        channel_stats = {}

        channel_data = self.data
        max_value = np.max(self.data).round(3)
        min_value = np.min(self.data).round(3)
        mean = np.mean(self.data).round(3)
        std_dev = np.std(self.data).round(3)
        rms = np.sqrt(np.mean(np.square(self.data))).round(3)
        dynamic_range = (max_value - min_value).round(3)  # Calculate dynamic range
        channel_stats = {
            stat_names[0]: max_value,
            stat_names[1]: min_value,
            stat_names[2]: mean,
            stat_names[3]: rms,
            stat_names[4]: dynamic_range
            }

        return channel_stats

    # Function to compute the average spectrum across a sample
    def average_spectrum(self, range=(0, 2000)):

        Audio_Object = Process.normalize(self)
        data = Audio_Object.data

        # Define the desired frequency range
        frequency_range = range
        spectrum = np.fft.fft(data)  # Apply FFT to the audio data
        magnitude = np.abs(spectrum)

        # Calculate frequency bins and positive frequency mask for each sample
        frequency_bins = np.fft.fftfreq(len(data), d=1 / self.SAMPLE_RATE)
        positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])

        channel_spectrums = [magnitude[positive_freq_mask][:len(frequency_bins)]]

        # Average across all channels
        average_spectrum = np.mean(channel_spectrums, axis=0)

        return average_spectrum, frequency_bins

    # Function to calculate spectrogram of audio
    def spectrogram(self, range=(0, 2000), stats=False):
        window_size = 32768
        hop_length = 512
        frequency_range = range

        Audio_Object = Process.normalize(self)
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
        self. freq_range_high = int(frequency_range[-1])
        self.freq_resolution = round(frequency_resolution, 2)


        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range: ({self.freq_range_low},{self. freq_range_high}) Hz')
            print(f'Freq Resolution: {self.freq_resolution} Hz')

        return spectrogram_db

    # Function to export an object
    def export(self, file_path):
        # Save/export the audio object
        sf.write(f'{file_path}', self.data, self.SAMPLE_RATE)