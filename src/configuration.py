"""
Utility class to hold the configurations
"""
from pydantic import BaseModel
from pathlib import Path

class GprMaxConfig(BaseModel):
    """
    Configuration for the GPR dataset creation script in create_dataset.py,
    the value types are automatically checked by pydantic.
    """
    n_samples: int
    generate_input: bool
    geometry_only: bool

    input_dir: Path
    output_dir: Path

    # simulation configuration
    n_ascans: int
    domain: tuple[float, float, float]
    spatial_resolution: tuple[float, float, float]
    delta_t: float

    source_waveform: str
    source_amplitude: float
    source_central_frequency: float
    source_position: tuple[float, float, float]
    receiver_position: tuple[float, float, float]
    step_size: tuple[float, float, float]

    fractal_dimension: float
    pep_soil_number: int

    materials: dict[str, tuple[float, ...]]

    layers: list[str]

    layer_sizes: tuple[float, ...]
    sleepers_separation: float
    sleepers_material: list[str]

    max_fouling_level: float
    max_fouling_water: float
    max_pss_water: float

