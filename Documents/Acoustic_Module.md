# Acoustic Module Documentation

The Acoustic module provides functionality for analyzing and processing audio files. It includes classes and functions for working with WAV files, handling metadata, visualizing audio data, and performing various audio processing tasks.

## Table of Contents

- [Acoustic Class](#acoustic-class)
- [Audio_MC Class](#audio_mc-class)
- [Process Functions](#process-functions)
- [Sample Library References](#sample-library-references)
- [Utility Functions](#utility-functions)
- [Visualization Functions](#visualization-functions)
- [Comparisons Module](#comparisons-module)
- [Mic_Mount Module](#mic-mount-module)

## Acoustic Class

The `Audio` class is used for analyzing WAV files. It provides methods for calculating statistics, generating spectrograms, and exporting audio data.

- `__init__(self, filepath, SR=Sample_Library.SAMPLE_LIBRARY_SAMPLE_RATE, stats=False)`: Initializes an instance of the `Audio` class with the specified WAV file. Optional parameters are available for sample rate (`SR`) and printing statistics (`stats`).
- `stats(self)`: Calculates statistics of the audio file and returns them as a dictionary.
- `average_spectrum(self, range=(0, 2000))`: Computes the average spectrum across a sample and returns the spectrum and corresponding frequency bins.
- `spectrogram(self, range=(0, 2000), stats=False)`: Calculates the spectrogram of the audio file and returns it as a numpy array. Optional parameters are available for frequency range, displaying statistics, and exporting the spectrogram as an image.
- `export(self, file_path)`: Exports the audio data to a WAV file.

## Audio_MC Class

The `Audio_MC` class is used for analyzing multi-channel WAV files. It provides methods for calculating statistics, visualizing waveforms, and generating spectrograms.

- `__init__(self, filepath, stats=False)`: Initializes an instance of the `Audio_MC` class with the specified multi-channel WAV file. Optional parameter `stats` indicates whether to print statistics about the audio file.
- `stats(self, display=False)`: Calculates statistics of the audio file and returns them as a list of dictionaries. Optional parameter `display` indicates whether to display the statistics as a table.
- `visualize(self, channel=1)`: Plots the waveform of the specified channel.
- `visualize_4ch(self)`: Plots the waveforms of all four channels.
- `spectro(self, channel=1, log=False, freq=(20, 2000))`: Plots the spectrogram of the specified channel. Optional parameters are available for logarithmic scale, frequency range, and exporting the spectrogram as an image.
- `spectro_4ch(self, log=False, freq=(20, 2000))`: Plots the spectrograms of all four channels.

## Process Functions

The `Process` module provides functions for processing audio data.

- `amplify(Audio_Object, gain_db)`: Amplifies or attenuates the audio data by the specified gain in decibels.
- `normalize(Audio_Object, percentage=95)`: Normalizes the audio data to the specified percentage of the maximum amplitude.

## Sample Library References

The `Samp_Lib` module contains references for a user-specific sample library.

- `SAMPLE_LIBRARY_DIRECTORY`: Path to the sample library directory.
- `SAMPLE_DIRECTORY`: Path to the sample directory within the sample library.
- `SAMPLE_LIBRARY_LIST`: Path to the CSV file containing sample metadata.
- `Samp_Lib` class: Provides methods for accessing and updating sample metadata.

## Utility Functions

The `Utils` module provides utility functions for working with files and directories.

- `create_directory_if_not_exists(file_path)`: Checks if a directory exists and creates it if it doesn't.
- `check_file_exists(file_path)`: Checks if a file exists.
- `copy_directory_structure(src_dir, dest_dir)`: Copies the directory structure from the source directory to the destination directory.

## Visualization Functions

The `Visualize` module provides functions for visualizing audio data.

- `waveform(Audio_Object)`: Plots the waveform of an audio object.
- `stats(Audio_Object)`: Plots the statistics (max, min, mean, RMS) of an audio object.
- `spectral_plot(Audio_Object)`: Plots the spectral plot of an audio object.
- `overview(Audio_Object, save=False, override=False, save_dir=None)`: Generates an overview of an audio object, including the waveform, statistics, and spectral plot. Optional parameters are available for saving the overview as a PDF file.

## Comparisons Module

The `Comparisons` module provides classes for comparing audio samples.

- `Compare` class: Provides methods for comparing audio samples.
- `Mount_Compare` class: Provides methods for comparing audio samples using microphone mounts.

## Mic_Mount Module

The `Mic_Mount` module defines classes for microphone mounts.

- `Mount` class: Represents a microphone mount with its characteristics.

## Entry Point and Usage

The Acoustic module can be used as an entry point for analyzing and processing audio files. It provides the following functionality:

- Importing and analyzing WAV files using the `Audio` and `Audio_MC` classes.
- Computing statistics, average spectrum, and spectrogram of audio data.
- Exporting audio objects to WAV files.
- Visualizing audio waveforms, statistics, and spectrograms.
- Working with a user-specific sample library for managing sample metadata.
- Performing random functions and processing audio data.

To use the Acoustic module, you can import it into your Python script as follows:

```python
from Acoustic import *
audio = Audio('path/to/audio.wav')
audio.waveform()
audio.stats()
audio.spectrogram()
