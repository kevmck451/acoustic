#Acoustic Main

from audio import Audio
import utils
from utils import CSVFile
import process
import visualize
import Sample_Library
from sample_library import Samp_Lib
from process import Process
from target import Target



def main():
    # Initialize Sample Library
    SampleLibrary = Samp_Lib()
    SampleLibrary.CSV.print_entries()

    filepath = '../Data/Sample Library/Samples/Originals/Full Flights/Hex_1_FullFlight_a.wav'
    filepath = '../Data/Sample Library/Samples/Originals/Takeoff/Angel_3_Takeoff_a.wav'

    sample = Audio(filepath, stats=False)

    visualize.overview(sample)
    visualize.spectrogram(sample)

    amp_sample = process.amplify(sample, 6)
    visualize.overview(amp_sample)

















if __name__ == '__main__':
    main()
