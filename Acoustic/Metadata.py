# Dataclasses for Metadata

from dataclasses import dataclass

@dataclass(frozen=True)
class environment:
    temperature: int
    relative_humidity: int
    barometric_pressure: int
    wind_speed: int
    wind_direction: str
    date: str
    time: str
    location: str
    vehicle: classmethod

@dataclass(frozen=True)
class conditions:
    coordinates: list
    altitude: list
    speed: list
    movement: str
    target: classmethod

@dataclass(frozen=True)
class sample:
    id: int
    directory: str
    filename: str
    category: classmethod
    sample_rate: int
    bit_depth: int
    channels: int
    channel_relation: str
    length: int
    quality_score: int

@dataclass(frozen=True)
class extracted:
    frequency_response: list
    temporal_characteristics: list
    spectral_characteristics: list
    spatial_characteristics: list

@dataclass(frozen=True)
class target:
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

