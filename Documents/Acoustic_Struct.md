# Acoustic Module Structure

1. **Audio Class**: Provides functionality for analyzing and processing single-channel WAV files. It includes methods for computing statistics, average spectrum, and spectrogram, as well as exporting audio data.

2. **Audio_MC Class**: Extends the functionality of the Audio class to support multi-channel audio files. It includes additional methods for visualizing multi-channel waveforms and spectrograms.

3. **Process Module**: Contains functions for audio processing tasks, such as amplifying and normalizing audio data. These functions can be used with both the Audio and Audio_MC classes.

4. **Sample_Library Module**: Handles the user-specific sample library and provides functions for accessing sample metadata.

5. **Utils Module**: Provides utility functions for working with files, directories, and other miscellaneous tasks.

6. **Visualize Module**: Includes functions for visualizing audio data, such as waveforms, statistics, and spectrograms.

