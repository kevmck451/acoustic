
from pathlib import Path
import soundfile as sf
import numpy as np
import wave


class Audio_Abstract:
    def __init__(self, **kwargs):
        stats = kwargs.get('stats', False)
        self.path = Path(kwargs.get('filepath', None))
        self.sample_rate = kwargs.get('sample_rate', 48000)
        self.num_channels = kwargs.get('num_channels', 1)
        self.sample_length = kwargs.get('sample_length', None)
        self.data = kwargs.get('data', None)
        self.num_samples = kwargs.get('num_samples', None)
        self.name = kwargs.get('name', self.path.stem)

        # If given a filepath
        if self.path is not None:
            with wave.open(str(self.path), 'rb') as wav_file:
                self.num_channels = wav_file.getnchannels()

            self.load_data(self.path)

        if stats:
            print(f'path: {self.path}')
            print(f'sample_rate: {self.sample_rate}')
            print(f'num_channels: {self.num_channels}')
            print(f'sample_length: {self.sample_length}')
            print(f'num_samples: {self.num_samples}')
            print(f'data type: {self.data.dtype}')
            print(f'data shape: {self.data.shape}')
            print(f'data: {self.data}')

    # Function that loads data from filepath
    def load_data(self, filepath):
        if self.num_channels > 1:
            self.data, _ = sf.read(str(filepath), dtype='float32')

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
            self.data, _ = sf.read(str(filepath), dtype='float32')
            self.sample_length = round((len(self.data) / self.sample_rate), 2)
            self.num_samples = len(self.data)

    # Function that returns stats from the audio file
    def stats(self):
        stat_names = ['Max', 'Min', 'Mean', 'RMS', 'Range']
        channel_stats = {}

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






if __name__ == '__main__':

    a = Audio_Abstract(stats=True)

    print('-' * 50)

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/mission 5/Hex_FullFlight_5.wav'
    b = Audio_Abstract(filepath=filepath, stats=True)

    print('-'*50)

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/Orlando/dataset 5/1/5_target_1_a.wav'
    c = Audio_Abstract(filepath=filepath, stats=True)





