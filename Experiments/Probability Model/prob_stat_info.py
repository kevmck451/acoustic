from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process
import audio_filepaths as af

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from pathlib import Path
import numpy as np
import librosa


# Noise Characterization: Stat Analysis: Freq Dist, Amp Var, Temp Char
# Probabilistic Modeling
# Spectral Subtraction

def freq_stats(directory):
    means, variances, skewnesses, kurtoses = [], [], [], []

    for filepath in Path(directory).rglob('*.wav'):
        audio = Audio_Abstract(filepath=filepath)
        audio = process.normalize(audio)
        spectrum, f_bins = process.average_spectrum(audio, norm=True)

        # Calculate statistics for this spectrum
        mean = np.mean(spectrum)
        variance = np.var(spectrum)
        skewness = skew(spectrum)
        kurt = kurtosis(spectrum)

        # Append statistics to the lists
        means.append(mean)
        variances.append(variance)
        skewnesses.append(skewness)
        kurtoses.append(kurt)

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(means, label='Mean')
    axs[0, 0].set_title('Mean of Spectra')
    axs[0, 1].plot(variances, label='Variance')
    axs[0, 1].set_title('Variance of Spectra')
    axs[1, 0].plot(skewnesses, label='Skewness')
    axs[1, 0].set_title('Skewness of Spectra')
    axs[1, 1].plot(kurtoses, label='Kurtosis')
    axs[1, 1].set_title('Kurtosis of Spectra')

    for ax in axs.flat:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # base = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover'
    # filepath = f'{base}/hex_hover_8_thin.wav'
    # filepath = f'{base}/Hex_8_Hover_3_d.wav'

    directory = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals/Hover')

    freq_stats(directory)
