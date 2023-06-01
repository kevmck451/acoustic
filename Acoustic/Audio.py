# Audio Class for Analyzing WAV Files
# Kevin McKenzie 2023

from pathlib import Path
import numpy as np
import librosa
import Sample_Library
from Utils import CSVFile


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



