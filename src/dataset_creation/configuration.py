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
    n_ascans: int
    generate_input: bool
    geometry_only: bool

    # folders
    input_dir: Path
    tmp_dir: Path
    output_dir: Path

    # simulation configuration
    domain: tuple[float, float, float]
    spatial_resolution: tuple[float, float, float]
    time_window: float

    track_configuration_probabilities: dict[str, float]

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
    layer_sizes: dict[str, tuple[float, float]]
    layer_roughness: dict[str, float]

    # Sleepers
    sleepers_separation: float
    sleepers_material_probabilities: dict[str, float]
    sleepers_sizes: dict[str, tuple[float, float, float]]

    # Fouling/water content
    fouling_box_threshold: float
    water_infiltration_threshold: float

    # Snapshots
    snapshot_times: list[float]