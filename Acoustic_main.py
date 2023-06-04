#Acoustic Main

from Audio import Audio
import Utils
from Utils import CSVFile
import Process
import Visualize
import Sample_Library
from Sample_Library import Samp_Lib



def main():
    # Initialize Sample Library
    # SampleLibrary = Samp_Lib()
    # SampleLibrary.CSV.print_entries()

    filepath = '../Data/Sample Library/Samples/Full Flights/Hex_1_FullFlight_a.wav'
    # sample = Audio(filepath, stats=False)

    # Visualize.overview(sample)
    # Visualize.spectrogram(sample)

    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the .wav file
    y, sr = librosa.load(filepath)

    # Compute the spectrogram of the data
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=32768)), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()























if __name__ == '__main__':
    main()
