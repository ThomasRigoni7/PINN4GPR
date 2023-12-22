"""
This module contains useful classes used to obtain metadata from the dataset generation processes.
"""


from pydantic import BaseModel
from typing import Literal, Optional
from pathlib import Path
import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt

class Metadata(BaseModel):  # numpydoc ignore=PR01
    """
    Utility class to keep the sample metadata information
    """

    seed: int

    track_type: Literal["PSS", "AC_rail", "subgrade"]
    
    ballast_simulation_seed: Optional[int]
    fouling_level: float
    is_fouled: bool
    general_deterioration: float
    
    layer_sizes: dict[str, float]

    general_water_content: float
    water_infiltrations: tuple[float, float, float]
    layer_water_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    sleepers_material : Literal["wood", "steel", "concrete"]
    sleeper_positions: list[tuple[float, float, float]]

    # calculated after the sampling, based on general deterioration 
    # and layer water ranges
    fouling_material: Optional[tuple[float, float]] = None
    pss_material: Optional[tuple[float, float]] = None
    subsoil_material: Optional[tuple[float, float]] = None


class DatasetStats():
    """
    Class used to store and save metadata from multiple runs to file.

    Parameters
    ----------
    stats : dict[str, Metadata]
        dictionary containing statistics relative to input files.
    """
    def __init__(self, stats: dict[str, Metadata]) -> None:
        self.stats = stats

    def write_metadata_files(self, directory: Path):
        """
        Writes on disk the dataset statistics.

        Parameters
        ----------
        directory : dict[str]
            Directory in which to save the dataset statistics.
        """
        # write a file for each sample
        directory.mkdir(exist_ok=True)
        for file, metadata in self.stats.items():
            info_file = (directory / file).with_suffix(".txt")
            with open(info_file, "w") as f:
                for k, j in metadata:
                    f.write(f"{k}: {j}\n")

        # write a single pkl file with all the informations
        with open(directory / "all_data.pkl", "wb") as f:
            pickle.dump(self, f)
        
        # calculate meaningful statistics for the dataset:
        track_type = {
            "PSS": 0,
            "AC_rail": 0,
            "subgrade": 0
        } # counts
        is_fouled = []      # percentage
        fouling_level = []  # distribution
        layer_sizes = {"fouling" : [], "ballast" : [], "asphalt" : [], "PSS" : []}    # distribution
        general_water_content = []  # distribution
        water_infiltrations = []    # percentage for each layer
        sleepers_material_counts = {"wood" : 0, "steel" : 0, "concrete": 0}      # percentage for each material
        water_contents = {"fouling": [], "PSS": [], "subsoil": []} # distribution for min and max for each layer
        sleeper_counts = [] # distribution 2 or 3 sleepers

        for file, metadata in self.stats.items():
            track_type[metadata.track_type] += 1
            is_fouled.append(metadata.is_fouled)
            fouling_level.append(metadata.fouling_level)
            for name, size in metadata.layer_sizes.items():
                layer_sizes[name].append(size)
            general_water_content.append(metadata.general_water_content)
            water_infiltrations.append(metadata.water_infiltrations)
            sleepers_material_counts[metadata.sleepers_material] += 1
            if metadata.is_fouled:
                water_contents["fouling"].append(metadata.fouling_material[4:])
            water_contents["PSS"].append(metadata.pss_material[4:])
            water_contents["subsoil"].append(metadata.subsoil_material[4:])
            sleeper_counts.append(len(metadata.sleeper_positions))
        
        is_fouled_percentage = np.array(is_fouled).mean()
        fouling_level = np.array(fouling_level)
        layer_sizes_distrib = {}
        for name, l in layer_sizes.items():
            layer_sizes_distrib[name] = np.array(l)
        general_water_content_distrib = np.array(general_water_content)
        water_infiltrations_percentages = np.array(water_infiltrations).mean(axis=0)
        layers_water_content_distrib = {}
        for name, data in water_contents.items():
            d = np.array(data)
            if d.ndim == 1:
                d = d[None, :]
            layers_water_content_distrib[name] = d
        sleepers_counts_distrib = collections.Counter(sleeper_counts)

        # write statistics
        with open(directory / "statistics.txt", "w") as f:
            f.write(f"track types: {track_type}\n")
            f.write(f"fouled percentage: {is_fouled_percentage}\n")
            f.write(f"water infiltrations percentages: {water_infiltrations_percentages}\n")
            f.write(f"sleeper number distribution: {sleepers_counts_distrib}\n")
            f.write(f"sleeper material distribution: {sleepers_material_counts}\n")

        # plot distributions
        plots_dir = directory / "plots"
        plots_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots()
        fig.suptitle("Fouling level")
        ax.hist(fouling_level, bins = 30)
        fig.savefig(plots_dir / "fouling_level.png")

        # layer sizes
        fig, axs = plt.subplots(ncols=4, sharey=True, tight_layout=True)
        fig.suptitle("layer sizes")
        axs[0].hist(layer_sizes_distrib["fouling"], bins = 30)
        axs[0].set_title("fouling")
        axs[1].hist(layer_sizes_distrib["ballast"], bins = 30)
        axs[1].set_title("ballast")
        axs[2].hist(layer_sizes_distrib["asphalt"], bins = 30)
        axs[2].set_title("asphalt")
        axs[3].hist(layer_sizes_distrib["PSS"], bins = 30)
        axs[3].set_title("PSS")
        fig.savefig(plots_dir / "layer_sizes.png")

        # general water content
        fig, ax = plt.subplots()
        fig.suptitle("General water content distribution")
        ax.hist(general_water_content_distrib, bins = 30)
        fig.savefig(plots_dir / "general_water_content.png")
        
        # layers water ranges
        fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, tight_layout = True)
        fig.suptitle("water content distributions")
        if is_fouled_percentage > 0.0:
            axs[0].hist2d(layers_water_content_distrib["fouling"][:, 0], layers_water_content_distrib["fouling"][:, 1], bins=30)
        axs[0].set_title("fouling")
        axs[1].hist2d(layers_water_content_distrib["PSS"][:, 0], layers_water_content_distrib["PSS"][:, 1], bins=30)
        axs[1].set_title("PSS")
        axs[2].hist2d(layers_water_content_distrib["subsoil"][:, 0], layers_water_content_distrib["subsoil"][:, 1], bins=30)
        axs[2].set_title("subsoil")
        fig.savefig(plots_dir / "water_content.png")
        plt.close("all")
