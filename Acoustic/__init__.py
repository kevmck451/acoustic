"""
Acoustic Module
===============
Author: Kevin McKenzie
Version: 1.0.0
Start Date: May 2023
===============
The Acoustic module provides a comprehensive set of tools for analyzing, processing, and visualizing audio data.

Classes:
- Audio: Analyzes and processes single-channel WAV files.
- Audio_MC: Extends Audio class to support multi-channel audio files.
- Samp_Lib: Manages user-specific sample library and metadata.
- Compare: Provides methods for comparing audio samples.
- Mount_Compare: Provides methods for comparing audio samples using microphone mounts.

Modules:
- Process: Contains functions for audio processing tasks.
- Sample_Library: Handles user-specific sample library and metadata.
- Utils: Provides utility functions for working with files and directories.
- Visualize: Includes functions for visualizing audio data.
- Comparisons: Contains classes for comparing audio samples.
- Mic_Mount: Defines classes for microphone mounts.
"""

from .Audio import Audio
from .Audio_MultiCh import Audio_MC
from .Sample_Library import Samp_Lib
from .Process import amplify, normalize
from .Sample_Library import (
    BASE_DIRECTORY,
    SAMPLE_LIBRARY_DIRECTORY,
    SAMPLE_DIRECTORY,
    SAMPLE_LIBRARY_LIST,
    ORIGINAL_DIRECTORY,
    NORMALIZED_DIRECTORY,
    OVERVIEW_DIRECTORY,
    OVERVIEW_ORIGINAL_DIRECTORY,
    OVERVIEW_NORMALIZE_DIRECTORY,
    SAMPLE_CATEGORIES,
    SAMPLE_HEADERS,
    SAMPLE_LIBRARY_SAMPLE_RATE,
    SAMPLE_LIBRARY_BIT_DEPTH,
)
from .Utils import (
    create_directory_if_not_exists,
    check_file_exists,
    copy_directory_structure,
    CSVFile,
)
from .Visualize import (
    waveform,
    stats,
    spectral_plot,
    overview,
    spectrogram,
)
from .Comparisons import Compare, Mount_Compare
from .Mic_Mount import Mount

__all__ = [
    "Audio",
    "Audio_MC",
    "Samp_Lib",
    "amplify",
    "normalize",
    "BASE_DIRECTORY",
    "SAMPLE_LIBRARY_DIRECTORY",
    "SAMPLE_DIRECTORY",
    "SAMPLE_LIBRARY_LIST",
    "ORIGINAL_DIRECTORY",
    "NORMALIZED_DIRECTORY",
    "OVERVIEW_DIRECTORY",
    "OVERVIEW_ORIGINAL_DIRECTORY",
    "OVERVIEW_NORMALIZE_DIRECTORY",
    "SAMPLE_CATEGORIES",
    "SAMPLE_HEADERS",
    "SAMPLE_LIBRARY_SAMPLE_RATE",
    "SAMPLE_LIBRARY_BIT_DEPTH",
    "create_directory_if_not_exists",
    "check_file_exists",
    "copy_directory_structure",
    "CSVFile",
    "waveform",
    "stats",
    "spectral_plot",
    "overview",
    "spectrogram",
    "Compare",
    "Mount_Compare",
    "Mount",
]
