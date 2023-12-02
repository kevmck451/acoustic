from audio_abstract import Audio_Abstract
import process



filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Experiments/Static Tests/Static Test 1/Samples/Tones/Noisy Signal/10_D_1000_M.wav'
audio = Audio_Abstract(filepath=filepath)
audio = process.normalize(audio)
# audio.waveform(display=True)



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

nperseg = 32768  # You can adjust this value

# Compute the spectrogram
frequencies, times, Sxx = spectrogram(audio.data, fs=audio.sample_rate, nperseg=nperseg)

# Convert to decibels
Sxx_dB = 10 * np.log10(Sxx)

plt.figure(figsize=(20, 6))

# Using a logarithmic scale for the y-axis (frequency)
plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
plt.yscale('log')  # Set the scale of the y-axis to logarithmic
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('High-Resolution Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.ylim([frequencies[1], frequencies[-1]])  # Avoid zero at log scale
plt.show()





