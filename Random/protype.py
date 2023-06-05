# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
#
# directory = '/Volumes/KM1TB/Orlando Files/Audio/Samples/'
# file_name = 'Hex_Flight_5.wav'
#
# file_path = directory + file_name
#
# CHANNEL_NUM = 4
# CHANNEL_INDICES = [0, 1, 2, 3]
# SAMPLE_RATE = 48000
#
# # Load the audio file
# audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
# # audio: NumPy array containing the audio samples
# # sr: Sampling rate of the audio file
# print(audio.shape)
#
# # Print the duration and sampling rate of the audio
# duration = librosa.get_duration(y=audio, sr=sr)
# print("Duration:", duration, "seconds")
# print("Sampling Rate:", sr, "Hz")
#
# # Check sample rate and change if necessary
# # librosa.resample()
# # Segmentation
# # Feature Extraction
# # Normalization
# # Data Augmentation
# # Data Splitting


# Load the audio sample
audio_file = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/' \
             '1 Acoustic/Data/Sample Library/Samples/Originals/' \
             'Full Flight/Hex_1_FullFlight_a.wav'

import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_waveforms(audio_path, frame_length=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Compute RMS value
    rms = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length)[0]

    # Prepare RMS signal for plotting
    rms_signal = np.zeros_like(y)
    for i, value in enumerate(rms):
        rms_signal[i * hop_length:(i + 1) * hop_length] = value

    # Plot original waveform and RMS waveform
    plt.figure(figsize=(14, 5))

    plt.subplot(2, 1, 1)
    plt.plot(y, label='Original Waveform')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rms_signal, label='RMS Waveform', color='red')
    plt.legend()

    plt.show()

# Test the function
plot_waveforms(audio_file)




