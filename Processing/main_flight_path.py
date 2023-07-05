#Acoustic Main

from audio import Audio
import utils
from utils import CSVFile
import process
import visualize
import Sample_Library
from sample_library import *
from sample_library import Samp_Lib
from process import Process
from target import Target
from flight_path import Flight_Path
from target import Target



def main():


    flight = Flight_Path(FLIGHT_LOG[0])
    # flight.plot_flight_path()
    # flight.display_target_distance(display=True)
    flight.label_flight_sections()











if __name__ == '__main__':
    main()
