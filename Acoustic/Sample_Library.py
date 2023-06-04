# References for the User Specific Sample Library

from pathlib import Path
import Utils


# If using this module, change these variables to match your own computer layout / metadata
SAMPLE_LIBRARY_DIRECTORY = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library'
SAMPLE_DIRECTORY = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Originals'
VISUALIZE_SAVE_DIRECTORY = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Overview/Originals'
SAMPLE_LIBRARY_LIST = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Info Files/SampleList.csv'

NORMALIZED_SAVE_DIRECTORY = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Sample Library/Samples/Normalized'

SAMPLE_CATEGORIES = ['Ambient', 'Flight', 'Flying', 'Full Flights', 'Hover', 'Hovering',
                     'Idle', 'Landings', 'Takeoffs', 'Vehicle Sounds']
SAMPLE_HEADERS = ['Sample', 'Location', 'Date', 'Time', 'Vehicle', 'Recorder', 'RAW', 'Category',
                  'Temp', 'Humidity', 'Pressure', 'Wind', 'Max', 'Min', 'Mean', 'RMS', 'Range']

SAMPLE_LIBRARY_SAMPLE_RATE = 48000
SAMPLE_LIBRARY_BIT_DEPTH = 16


class Samp_Lib:
    def __init__(self):
        self.SampLib = Path(SAMPLE_LIBRARY_DIRECTORY)

        # Initialize the Sample Library CSV File
        self.CSV = Utils.CSVFile(SAMPLE_LIBRARY_LIST)

        print('Sample Library Initialized')

