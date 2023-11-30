"""
Utility class to hold the configurations
"""
from pydantic import BaseModel
from pathlib import Path

class GprMaxConfig(BaseModel):  # numpydoc ignore=PR01
    """
    Utility class to hold the configuration gprMax configuration to generate a dataset.
    """
    # general config
    n_samples: int
    generate_input: bool
    geometry_only: bool

    # folders
    input_dir: Path
    tmp_dir: Path
    output_dir: Path

    # simulation configuration
    n_ascans: int
    domain: tuple[float, float, float]
    spatial_resolution: tuple[float, float, float]
    time_window: float

    # source/receiver
    source_waveform: str
    source_amplitude: float
    source_central_frequency: float
    source_position: tuple[float, float, float]
    receiver_position: tuple[float, float, float]
    step_size: tuple[float, float, float]

    # Peplinski soils
    fractal_dimension: float
    pep_soil_number: int

    # Materials and layers
    materials: dict[str, tuple[float, ...]]
    antenna_sleeper_distance: float
    layer_sizes: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    # Sleepers
    sleepers_separation: float
    sleepers_material: list[str]
    sleepers_size: tuple[float, float, float]

    # Fouling/water content
    max_fouling_percentage: float
    fouling_water_range: tuple[float, float]
    pss_water_range: tuple[float, float]

    # Snapshots
    snapshot_times: list[float]