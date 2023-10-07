# Functions to Process Audio

from Acoustic.audio_abstract import Audio_Abstract
from Acoustic import utils

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import signal
import numpy as np
import librosa
from pydub import AudioSegment




#-----------------------------------
# FEATURES -------------------------
#-----------------------------------
# Function to calculate spectrogram of audio (Features are 2D)
def spectrogram(audio_object, **kwargs):
    stats = kwargs.get('stats', False)
    range = kwargs.get('feature_params', 'None')
    window_sizes = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 254]
    window_size = window_sizes[2]
    hop_length = 512

    data = audio_object.data
    # Audio_Object = normalize(audio_object)
    # data = Audio_Object.data

    # Initialize an empty list to store the spectrograms for each channel
    spectrograms = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = np.abs(librosa.stft(channel_data, n_fft=window_size, hop_length=hop_length)) ** 2

        # Convert to decibels (log scale) for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Apply Min-Max normalization to the spectrogram_db
        spectrogram_db_min, spectrogram_db_max = spectrogram_db.min(), spectrogram_db.max()
        spectrogram_db = (spectrogram_db - spectrogram_db_min) / (spectrogram_db_max - spectrogram_db_min)

        # print(spectrogram_db_min)
        # print(spectrogram_db_max)
        # print(spectrogram_db)

        # Calculate frequency range and resolution
        nyquist_frequency = audio_object.sample_rate / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)
        frequency_range = np.arange(0, window_size // 2 + 1) * frequency_resolution

        if range == 'None':
            range = (70, 6000)

        bottom_index = int(np.round(range[0] / frequency_resolution))
        top_index = int(np.round(range[1] / frequency_resolution))

        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range: ({range[0]},{range[1]}) Hz')
            print(f'Freq Resolution: {frequency_resolution} Hz')

        # Cut the spectrogram to the desired frequency range and append to the list
        spectrograms.append(spectrogram_db[bottom_index:top_index])

    # Convert the list of spectrograms to a numpy array and return
    return np.array(spectrograms)

# Function to calculate MFCC of audio (Features are 2D)
def mfcc(audio_object, **kwargs):
    stats = kwargs.get('stats', False)
    n_mfcc = kwargs.get('feature_params', 'None')

    data = audio_object.data
    # Normalize audio data
    # Audio_Object = normalize(audio_object)
    # data = Audio_Object.data

    # Initialize an empty list to store the MFCCs for each channel
    mfccs_all_channels = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate MFCCs for this channel
        if n_mfcc == 'None':
            n_mfcc = 50
        mfccs = librosa.feature.mfcc(y=channel_data, sr=audio_object.sample_rate, n_mfcc=n_mfcc)

        # Normalize the MFCCs
        mfccs = StandardScaler().fit_transform(mfccs)

        if stats:
            print(f'MFCC: {mfccs}')

        # Append to the list
        mfccs_all_channels.append(mfccs)

    # Convert the list of MFCCs to a numpy array and return
    return np.array(mfccs_all_channels)

# Function to calculate spectrogram of audio (Features are 2D)
def custom_filter_1(audio_object, **kwargs):
    stats = kwargs.get('stats', False)
    window_size = 32768
    hop_length = 512
    # freq_range_low = (100, 170) #if hovering only
    freq_range_mid = (130, 200)
    # freq_range_high = (2600, 5000)
    freq_range_high = (800, 1300)

    data = audio_object.data
    # Audio_Object = normalize(audio_object)
    # data = Audio_Object.data

    # Initialize an empty list to store the spectrograms for each channel
    spectrograms = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        spectrogram = np.abs(librosa.stft(channel_data, n_fft=window_size, hop_length=hop_length)) ** 2

        # Convert to decibels (log scale) for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Apply Min-Max normalization to the spectrogram_db
        spectrogram_db_min, spectrogram_db_max = spectrogram_db.min(), spectrogram_db.max()
        spectrogram_db = (spectrogram_db - spectrogram_db_min) / (spectrogram_db_max - spectrogram_db_min)

        # Calculate frequency range and resolution
        nyquist_frequency = audio_object.sample_rate / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)
        frequency_range = np.arange(0, window_size // 2 + 1) * frequency_resolution

        bottom_index_mid = int(np.round(freq_range_mid[0] / frequency_resolution))
        top_index_mid = int(np.round(freq_range_mid[1] / frequency_resolution))
        bottom_index_high = int(np.round(freq_range_high[0] / frequency_resolution))
        top_index_high = int(np.round(freq_range_high[1] / frequency_resolution))

        spectrogram_mid = spectrogram_db[bottom_index_mid:top_index_mid]
        spectrogram_high = spectrogram_db[bottom_index_high:top_index_high]

        # Combine the 'mid' and 'high' spectrograms and append to the list
        spectrograms.append(np.concatenate((spectrogram_mid, spectrogram_high)))

        if stats:
            print(f'Spectro_dB: {spectrogram_db}')
            print(f'Freq Range Mid: ({freq_range_mid[0]},{freq_range_mid[1]}) Hz')
            print(f'Freq Range High: ({freq_range_high[0]},{freq_range_high[1]}) Hz')
            print(f'Freq Resolution: {frequency_resolution} Hz')

    # Convert the list of spectrograms to a numpy array and return
    return np.array(spectrograms)

# Function to calculate Zero Crossing Rate of audio (Features are 1D)
def zcr(audio_object, **kwargs):
    def sliding_window_reshape(data, window_size, step_size=1):
        """Reshape data using a sliding window."""
        num_windows = (len(data) - window_size) // step_size + 1
        output = np.zeros((num_windows, window_size))

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            output[i] = data[start:end]

        return output

    stats = kwargs.get('stats', False)

    data = audio_object.data
    # Normalize audio data
    # Audio_Object = normalize(audio_object)
    # data = Audio_Object.data

    # Initialize an empty list to store the ZCRs for each channel
    zcr_all_channels = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate ZCR for this channel
        window_size = 4096 # 8192
        hop_length = 1024 # 2048
        zcr_values = librosa.feature.zero_crossing_rate(y=channel_data,
                                                        frame_length=window_size,
                                                        hop_length=hop_length)

        # Normalize the ZCR values
        zcr_values = StandardScaler().fit_transform(zcr_values.reshape(-1, 1)).flatten()

        if stats:
            print(f'ZCR: {zcr_values}')

        # Reshape the ZCR using sliding window
        window_size = 3
        step_size = 1
        zcr_2d = sliding_window_reshape(zcr_values, window_size, step_size)

        # Now the shape of zcr_2d is [number_of_windows, window_size], add the channel dimension
        zcr_all_channels.append(zcr_2d[:, :, np.newaxis])

    # Convert the list of ZCRs to a numpy array and return
    return np.array(zcr_all_channels)

#-----------------------------------
# OTHER ----------------------------
#-----------------------------------
# Function to Normalize Data
def takeoff_trim(audio_object, takeoff_time):
    audio_copy = deepcopy(audio_object)
    samples_to_remove = int(np.round((takeoff_time * audio_object.sample_rate) + 1))
    audio_copy.data = audio_object.data[:, samples_to_remove:]

    return audio_copy

# Function to window over a sample of a specific length
def generate_windowed_chunks(audio_object, window_size, training=False):
    window_samples = audio_object.sample_rate * window_size
    half_window_samples = window_samples // 2
    total_samples = len(audio_object.data)

    audio_ob_list = []
    labels = []

    for window_start in range(0, total_samples, audio_object.sample_rate):
        audio_copy = deepcopy(audio_object)
        if window_start < half_window_samples:
            start = 0
            end = window_samples
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)
        elif window_start >= total_samples - half_window_samples:
            start = total_samples - window_samples
            end = total_samples
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)
        else:
            start = window_start - half_window_samples
            end = start + window_samples
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)
        audio_copy.data = audio_object.data[start:end]
        audio_ob_list.append(audio_copy)

    if training:
        if len(audio_ob_list) != len(labels):
            print(f'Error: {audio_object.path.stem}')
        return audio_ob_list, labels
    else: return audio_ob_list

# Function to convert audio sample to a specific length
def generate_chunks(audio_object, length, training=False):

    num_samples = audio_object.sample_rate * length
    start = 0
    end = num_samples
    total_samples = len(audio_object.data)

    audio_ob_list = []
    labels = []

    # If the audio file is too short, pad it with zeroes
    if total_samples < num_samples:
        audio_object.data = np.pad(audio_object.data, (0, num_samples - len(audio_object.data)))
        audio_object.sample_length = length
        audio_object.num_samples = length * audio_object.sample_rate
        audio_ob_list.append(audio_object)
        if training:
            label = int(audio_object.path.parent.stem)
            labels.append(label)  # Add Label (folder name)
    # If the audio file is too long, shorten it

    else:
        while end <= total_samples:
            audio_copy = deepcopy(audio_object)
            audio_copy.data = audio_object.data[start:end]
            audio_copy.sample_length = length
            audio_copy.num_samples = length * audio_copy.sample_rate
            audio_ob_list.append(audio_copy)
            start, end = (start + num_samples), (end + num_samples)
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)

    if training:
        if len(audio_ob_list) != len(labels):
            print(f'Error: {audio_object.path.stem}')
        return audio_ob_list, labels
    else: return audio_ob_list

# Function to convert audio sample to a specific length
def generate_chunks_4ch(audio_object, length, training=False):
    num_samples = audio_object.sample_rate * length
    start = 0
    end = num_samples
    total_samples = audio_object.data.shape[1]

    audio_ob_list = []
    labels = []

    # If the audio file is too short, pad it with zeroes
    if total_samples < num_samples:
        audio_object.data = np.pad(audio_object.data, (0, num_samples - len(audio_object.data)))
        audio_ob_list.append(audio_object)
        if training:
            label = int(audio_object.path.parent.stem)
            labels.append(label)  # Add Label (folder name)
    # If the audio file is too long, shorten it

    elif total_samples > num_samples:
        while end <= total_samples:
            audio_copy = deepcopy(audio_object)
            audio_copy.data = audio_object.data[:, start:end]
            audio_ob_list.append(audio_copy)
            start, end = (start + num_samples), (end + num_samples)
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)

    if training:
        if len(audio_ob_list) != len(labels):
            print(f'Error: {audio_object.path.stem}')
        return audio_ob_list, labels
    else: return audio_ob_list

# Function to convert 4 channel wav to list of 4 objects
def channel_to_objects(audio_object):

    if audio_object.num_channels == 4:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]
        audio_c = deepcopy(audio_object)
        audio_c.data = audio_object.data[2]
        audio_d = deepcopy(audio_object)
        audio_d.data = audio_object.data[3]

        return [audio_a, audio_b, audio_c, audio_d]

    elif audio_object.num_channels == 3:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]
        audio_c = deepcopy(audio_object)
        audio_c.data = audio_object.data[2]

        return [audio_a, audio_b, audio_c]

    else:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]

        return [audio_a, audio_b]

# Function to calculate the Signal to Noise Ratio with PSD
def signal_noise_ratio_psd(signal, noise, ):
    frequencies, psd_signal = power_spectral_density(signal)
    frequencies , psd_noise = power_spectral_density(noise)

    snr = 10 * np.log10(psd_signal / psd_noise)
    return frequencies, snr

# Function to calculate the Signal to Noise Ratio with RMS
def signal_noise_ratio_rms(signal, noise):
    rms_sig = root_mean_square(signal)
    rms_noise = root_mean_square(noise)
    snr = 10 * np.log10(rms_sig / rms_noise)
    return snr

# Function to calculate the Power Spectral Density
def power_spectral_density(audio_object):
    # freq is list of frequencies and psd is array with number of channels
    # that contains the power at those frequencies power (watts) per Hz

    frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate/4, nfft=32768) # 2048, 4096, 8192, 16384
    # frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate, average='mean') # 2048, 4096, 8192, 16384


    from matplotlib import pyplot as plt
    import pandas as pd

    avg_freq, avg_data = signal.welch(x=audio_object.data[0], fs=audio_object.sample_rate/4, average='mean')
    avg_data_pd = pd.Series(avg_data).rolling(13, center=True).mean().to_numpy()
    plt.figure(1, figsize=(14,8)).clf()
    plt.semilogy(avg_freq, avg_data, label='Raw PSD', lw=1, alpha=0.75)
    plt.semilogy(avg_freq, avg_data_pd, label='Rolled', lw=1, alpha=0.75)
    plt.title('\nPower Spectral Density Estimate')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power Spectral Density $\left(\frac{V^{2}}{Hz}\right)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return frequencies, psd

# Function to calculate the RMS Power
def root_mean_square(audio_object):
    rms = np.sqrt(np.mean(np.square(audio_object.data)))

    return rms

# Function to Increase or Decrease Sample Gain
def amplify(audio_object, gain_db):
    Audio_Object_amp = deepcopy(audio_object)

    # convert gain from decibels to linear scale
    gain_linear = 10 ** (gain_db / 20)

    # multiply the audio data by the gain factor
    Audio_Object_amp.data *= gain_linear

    return Audio_Object_amp

# Function to get average spectral values
def average_spectrum(audio_object, **kwargs):
    frequency_range = kwargs.get('frequency_range', (0, 20000))
    data = audio_object.data
    spectrum = np.fft.fft(data)  # Apply FFT to the audio data
    magnitude = np.abs(spectrum)
    frequency_bins = np.fft.fftfreq(len(data), d=1 / audio_object.sample_rate)
    positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])
    channel_spectrums = [magnitude[positive_freq_mask][:len(frequency_bins)]]
    average_spectrum = np.mean(channel_spectrums, axis=0)

    return average_spectrum, frequency_bins

# Function to mix down multiple channels to mono
def mix_to_mono(*audio_objects):
    # Check if audio objects have the same sample rate
    sample_rate = audio_objects[0].sample_rate
    for audio in audio_objects:
        if audio.sample_rate != sample_rate:
            raise ValueError("All audio objects must have the same sample rate!")

    # Initialize the combined data as zeros, using the maximum number of samples from the audio objects
    max_samples = max([audio.num_samples for audio in audio_objects])
    combined_data = np.zeros(max_samples)

    # Iterate through each audio object and sum up the data
    for audio in audio_objects:
        # If the audio has multiple channels, sum them to mono first
        if audio.num_channels > 1:
            if len(audio.data.shape) == 1:  # if audio.data is 1D
                mono_data = audio.data
            else:  # if audio.data is multi-dimensional
                mono_data = np.mean(audio.data, axis=0)
            # Ensure that the mono_data length matches the combined_data
            if len(mono_data) < len(combined_data):
                padding = np.zeros(len(combined_data) - len(mono_data))
                mono_data = np.concatenate((mono_data, padding))
            combined_data += mono_data
        else:
            # Ensure that the audio.data length matches the combined_data
            if len(audio.data) < len(combined_data):
                padding = np.zeros(len(combined_data) - len(audio.data))
                combined_data += np.concatenate((audio.data, padding))
            else:
                combined_data += audio.data

    # Average the combined data
    combined_data /= len(audio_objects)

    # Create a new Audio_Abstract object with the combined data
    mixed_audio = Audio_Abstract(data=combined_data, sample_rate=sample_rate, num_channels=1,
                                 num_samples=len(combined_data))

    return mixed_audio

#-----------------------------------
# PREPROCESSING --------------------
#-----------------------------------
# Function to Normalize Data
def normalize(audio_object, percentage=95):
    # make a deep copy of the audio object to preserve the original
    audio_normalized = deepcopy(audio_object)
    max_value = np.max(np.abs(audio_normalized.data))
    normalized_data = audio_normalized.data / max_value * (percentage / 100.0)

    audio_normalized.data = normalized_data

    return audio_normalized

# Function to compress audio
def compression(audio_object, threshold=-20, ratio=3.0, gain=1, attack=5, release=40):
    # Extracting the audio data
    audio_data = audio_object.data

    # Ensure audio_data is mono for simplicity
    if audio_object.num_channels > 1:
        audio_data = np.mean(audio_data, axis=0)

    # Convert dB threshold to amplitude
    threshold_amplitude = librosa.db_to_amplitude(threshold)

    # Apply compression
    compressed_data = np.zeros_like(audio_data)
    for i, sample in enumerate(audio_data):
        if abs(sample) > threshold_amplitude:
            gain_dB = librosa.amplitude_to_db(np.array([abs(sample)]))[0]
            gain = (gain_dB - threshold) / ratio
        else:
            gain = 1.0

        # Attack/release dynamics (basic form)
        target_gain = min(gain, 1.0)
        step = (target_gain - gain) / (attack if target_gain < gain else release)
        gain += step

        compressed_data[i] = sample * gain

    # Return a new Audio_Abstract object with the compressed data
    compressed_audio = Audio_Abstract(data=compressed_data, sample_rate=audio_object.sample_rate, num_channels=1,
                                      num_samples=len(compressed_data))

    return compressed_audio

# Function to subtract Hex from sample
def spectra_subtraction_hex(audio_object, **kwargs):
    pass
















class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)




if __name__ == '__main__':

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Static Tests/Static Test 1/Samples/Engine_1/3_10m-D-DEIdle.wav'
    audio = Audio_Abstract(filepath=filepath)
    audio.data = audio.data[2]
    audio = normalize(audio)
    # print(audio)

    # feature = zcr(audio, stats=False)
    # print(feature.shape)
    #
    # feature = spectrogram(audio)
    # print(feature.shape)
    #
    # feature = mfcc(audio)
    # print(feature.shape)

    spectra_subtraction_hex(audio)