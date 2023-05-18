from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

# wave_file_path = 'Audio Files/Low 1-6k Cut.wav'
# sr, signal = wavfile.read(wave_file_path)

# t = np.arange(len(y)) / float(sr)


wave_file_path = 'Audio Files/Low 1-6k Cut.wav'
sr1, signal1 = wavfile.read(wave_file_path)
y = signal1[:]  # use the first channel (or take their average, alternatively)
frq1, X1 = frequency_spectrum(y, sr1)

plt.figure()
plt.title('')
plt.subplot(2, 1, 1)
plt.plot(frq1, X1, 'b')
plt.xlabel('t')
plt.ylabel('y')

wave_file_path = 'Audio Files/Low 1-6k Boost.wav'
sr2, signal2 = wavfile.read(wave_file_path)
y = signal2[:]  # use the first channel (or take their average, alternatively)
frq2, X2 = frequency_spectrum(y, sr2)

plt.subplot(2, 1, 2)
plt.plot(frq2, X2, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.tight_layout()

plt.show()