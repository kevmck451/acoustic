# Sample Class for Analyzing Flight Data
# Kevin McKenzie 2023

from Acoustic_1 import*
from dataclasses import dataclass

class Sample:
    sample_library_base_directory = '../Data/Sample Library/'

    def __init__(self):
        pass

        # if base directory folders dont exist, create them
            # Info Files / Samples
        # Space
        # Environment
        # Targets
        # Noises
        # Position

        # sample_gain = Audio(file, False)






@dataclass(frozen=True)
class environment:
    temperature: int
    relative_humidity: int
    barometric_pressure: int
    wind_speed: int
    # wind_direction:

@dataclass(frozen=True)
class sound_source:
    type: str
    subtype: str
    name: str
    spectral_signature: dict
    temporal_signature: dict
    spatial_signature: dict
    # intensity:

@dataclass(frozen=True)
class microphones:
    type: str
    subtype: str
    name: str
    model_number: str
    acoustic_overload_point: int
    frequency_response: dict
    frequency_response_correction: dict
    # field_of_view:
    # sensitivity:
