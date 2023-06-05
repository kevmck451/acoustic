#Acoustic Main

from Audio import Audio
import Utils
from Utils import CSVFile
import Process
import Visualize
import Sample_Library
from Sample_Library import Samp_Lib
from Process import Process



def main():
    # Initialize Sample Library
    # SampleLibrary = Samp_Lib()
    # SampleLibrary.CSV.print_entries()

    # filepath = '../Data/Sample Library/Samples/Originals/Full Flights/Hex_1_FullFlight_a.wav'
    filepath = '../Data/Sample Library/Samples/Originals/Takeoffs/Angel_3_Takeoff_a.wav'

    sample = Audio(filepath, stats=False)

    # Visualize.overview(sample)
    # Visualize.spectrogram(sample)

    amp_sample = Process.amplify(sample, 6)
    Visualize.overview(amp_sample)
























if __name__ == '__main__':
    main()
