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
    SampleLibrary = Samp_Lib()
    # SampleLibrary.CSV.print_entries()

    filepath = '../Data/Sample Library/Samples/Flight/Source/Hex_3_Flight_a.wav'
    sample = Audio(filepath, stats=True)

    # Visualize.overview(sample)


























if __name__ == '__main__':
    main()
