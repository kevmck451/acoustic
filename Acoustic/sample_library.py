# References for the User Specific Sample Library

from pathlib import Path
import utils


# If using this module, change these variables to match your own computer layout / metadata
BASE_DIRECTORY = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic'

SAMPLE_LIBRARY_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library'
SAMPLE_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Samples'
SAMPLE_LIBRARY_LIST = BASE_DIRECTORY + '/Data/Sample Library/Info Files/SampleList.csv'

ORIGINAL_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Samples/Originals'
NORMALIZED_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Samples/Normalized'

OVERVIEW_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Overview'
OVERVIEW_ORIGINAL_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Overview/Originals'
OVERVIEW_NORMALIZE_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Overview/Normalized'
FLIGHT_PATH_SAVE_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Overview/Flights Paths'
TARGET_DISTANCE_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Overview/Target Distance'

TARGET_FLIGHT_DIRECTORY = BASE_DIRECTORY + '/Data/Sample Library/Info Files/Flights'


SAMPLE_CATEGORIES = ['Ambient', 'Flight', 'Flying', 'Full Flights', 'Hover',
                     'Idle', 'Landings', 'Takeoffs', 'Vehicle Sounds']
SAMPLE_HEADERS = ['Sample', 'Location', 'Date', 'Time', 'Vehicle', 'Recorder', 'RAW', 'Category',
                  'Temp', 'Humidity', 'Pressure', 'Wind', 'Max', 'Min', 'Mean', 'RMS', 'Range']

SAMPLE_LIBRARY_SAMPLE_RATE = 48000
SAMPLE_LIBRARY_BIT_DEPTH = 16


class Samp_Lib:
    def __init__(self):
        self.SampLib = Path(SAMPLE_LIBRARY_DIRECTORY)

        # Initialize the Sample Library CSV File
        self.CSV = utils.CSVFile(SAMPLE_LIBRARY_LIST)

        print('Sample Library Initialized')


