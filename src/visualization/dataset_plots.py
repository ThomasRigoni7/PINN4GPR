from pathlib import Path
from pickle import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from distutils.dir_util import copy_tree

from src.dataset_creation.statistics import DatasetStats, Metadata

def find_samples(dataset_path: Path):
    """
    Searches in the dataset for some input files and geometries, suitable for showing the diverseness of the dataset.
    """
    
    statistics_file = dataset_path / "gprmax_input_files/metadata/all_data.pkl"

    with open(statistics_file, "rb") as f:
        dataset_stats: DatasetStats = load(f)

    wanted_samples = [
        {
            "type": "PSS",
            "sleepers": "wood",
            "ballast": "clean",
            "deterioration": "new",
            "water": "dry",
            "infiltrations": (False, False, False)
        },
        {
            "type": "AC_rail",
            "sleepers": "concrete",
            "ballast": "clean",
            "deterioration": "old",
            "water": "dry",
            "infiltrations": (False, False, False)
        },
        {
            "type": "subgrade",
            "sleepers": "steel",
            "ballast": "fouled",
            "deterioration": "new",
            "water": "mid",
            "infiltrations": (False, False, False)
        },
        {
            "type": "AC_rail",
            "sleepers": "concrete",
            "ballast": "clean",
            "deterioration": "new",
            "water": "wet",
            "infiltrations": (True, True, True)
        },
    ]
    found = [False] * len(wanted_samples)
    samples = [None] * len(wanted_samples)

    for name, data in dataset_stats.stats.items():
        for i, wanted in enumerate(wanted_samples):
            if found[i] == True:
                continue
            if data.track_type != wanted["type"]:
                continue
            if data.sleepers_material != wanted["sleepers"]:
                continue

            if wanted["ballast"] == "clean":
                if data.fouling_level > 0.25:
                    continue
            elif wanted["ballast"] == "mid":
                if data.fouling_level < 0.25 or data.fouling_level > 0.8:
                    continue   
            elif wanted["ballast"] == "fouled":
                if data.fouling_level < 0.8:
                    continue

            if wanted["deterioration"] == "new":
                if data.general_deterioration > 0.25:
                    continue
            elif wanted["deterioration"] == "old":
                if data.general_deterioration < 0.8:
                    continue

            if wanted["water"] == "dry":
                if data.general_water_content > 0.25:
                    continue
            if wanted["water"] == "mid":
                if data.general_water_content < 0.25 or data.general_water_content > 0.8:
                    continue
            if wanted["water"] == "wet":
                if data.general_water_content < 0.8:
                    continue
            
            if wanted["infiltrations"] != data.water_infiltrations:
                continue

            print(i, name)
            found[i] = True
            samples[i] = name, data

    return samples

def copy_samples(dataset_path: Path, destination: Path, samples: list[tuple[str, Metadata]]):
    output_folder = dataset_path / "gprmax_output_files"
    for sample_name, _ in samples:
        sample_folder = output_folder / sample_name
        dst_folder = destination / sample_name
        copy_tree(str(sample_folder), str(dst_folder))

def visualize_samples(dataset_path: Path, samples: list[tuple[str, Metadata]]):
    output_folder = dataset_path / "gprmax_output_files"
    figures_folder = Path("figures")
    for sample_name, _ in samples:
        sample_folder = output_folder / sample_name
        sample_geometry = sample_folder / (sample_name + "_geometry.npy")

        geometry = np.load(sample_geometry)
        print(sample_name)
        plt.imshow(geometry[0], vmin=1, vmax=20, cmap="jet")
        plt.imsave((figures_folder / sample_name).with_suffix(".png"), geometry[0], vmin=1, vmax=20, cmap="jet")
        plt.colorbar()
        plt.show()

dataset_path = Path("dataset")
samples = find_samples(dataset_path)
copy_samples(dataset_path, Path("dataset_plots/output"), samples)
visualize_samples(dataset_path, samples)