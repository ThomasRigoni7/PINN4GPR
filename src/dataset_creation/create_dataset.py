"""
This file is used to create a GPR dataset using gprMax.

First creates some input files based on the provided configuration, then runs gprMax on the input files to get the output.

It also creates geometry files and converts their content into numpy arrays containing the relevant physical values.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from yaml import safe_load
import pickle
import numpy as np
import collections
import matplotlib.pyplot as plt

from .convert_to_np import convert_geometry_to_np, convert_snapshots_to_np
from .inputfile import InputFile
from .configuration import GprMaxConfig


def _parse_arguments():
    """
    Parses the arguments and returns the derived Namespace.
    """
    
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("-n_samples", type=int, help="Number of input files to generate/simulations to run.")
    parser.add_argument("-n_ascans", type=int,
                        help="Number of A-scans that constitute a B-scan, default=55")
    parser.add_argument("-generate_input", action="store_true", 
                        help="If True, generate input files. Otherwise use the files inside '-input_dir'.")
    parser.add_argument("-input_dir", type=str,
                        help="Directory to put the generated input files.")
    parser.add_argument("-output_dir", type=str,
                        help="Directory to store the generated results.")
    parser.add_argument("-geometry_only", action="store_true",
                        help="If set, only generate the geometries corresponding to the input files, but don't run the simulations.")
    parser.add_argument("-gprmax_config", type=str, default="gprmax_config.yaml",
                        help="Path to the gprmax yaml config file.")
    
    # simulation settings
    parser.add_argument("-layer_sizes", nargs=4, type=float,
                        help="Sizes of the gravel/asphalt/pss/ballast layers. Interpreted as cumulative height.")
    parser.add_argument("-sleepers_separation", type=float,
                        help="Separation between the sleepers in meters.")
    parser.add_argument("-sleepers_material", nargs="*", type=str, choices=["all", "steel", "concrete", "wood"])
    parser.add_argument("-max_fouling_level", type=float,
                        help="Maximum ballast fouling height in meters, measured from the pss layer interface.")
    parser.add_argument("-max_fouling_water", type=float,
                        help="Maximum percentage of water in fouling material between ballast stones. Default 0.15 means 15%%.")
    parser.add_argument("-max_pss_water", type=float,
                        help="Maximum percentage of water in the pss material. Default 0.15 means 15%%.")
    
    args = parser.parse_args()
    return args
    
def _resolve_directories(config: GprMaxConfig):
    """
    Resolves and creates the input, tmp and output directories.
    """
    config.input_dir = config.input_dir.resolve()
    config.tmp_dir = config.tmp_dir.resolve()
    config.output_dir = config.output_dir.resolve()
    config.input_dir.mkdir(exist_ok=True, parents=True)
    config.tmp_dir.mkdir(exist_ok=True, parents=True)
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
def _write_metadata_files(metadata: dict[str], metadata_dir: Path):
    """
    Creates the metadata files from the inputfiles metadata in create_gprmax_input_files().

    Parameters
    ----------
    metadata : dict[str]
        informations about the sampled values of the inputfiles
    """
    # write a file for each sample
    metadata_dir.mkdir(exist_ok=True)
    for file, i in metadata.items():
        info_file = (metadata_dir / file).with_suffix(".txt")
        with open(info_file, "w") as f:
            for k, j in i.items():
                f.write(f"{k}: {j}\n")

    # write a single pkl file with all the informations
    with open(metadata_dir / "all_data.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    # calculate meaningful statistics for the dataset:
    AC_rail = []        # percentage
    is_fouled = []      # percentage
    layer_sizes = {"fouling" : [], "ballast" : [], "asphalt" : [], "PSS" : []}    # distribution
    general_water_content = []  # distribution
    water_infiltrations = []    # pergentage for each layer
    sleepers_material_counts = {"wood" : 0, "steel" : 0, "concrete": 0}      # percentage for each material
    water_contents = {"fouling": [], "PSS": [], "subsoil": []} # distribution for min and max for each layer
    sleeper_counts = [] # distribution 2 or 3 sleepers

    for file, info in metadata.items():
        AC_rail.append(info["AC rail"])
        is_fouled.append(info["is fouled"])
        for name, size in info["layer sizes"].items():
            layer_sizes[name].append(size)
        general_water_content.append(info["general water content"])
        water_infiltrations.append(info["water infiltrations"])
        mat = info["sleepers material"]
        sleepers_material_counts[mat] += 1
        if "fouling water" in info:
            water_contents["fouling"].append(info["fouling water"])
        water_contents["PSS"].append(info["pss water"])
        water_contents["subsoil"].append(info["subsoil water"])
        sleeper_counts.append(len(info["sleeper positions"]))
    
    AC_rail_percentage = np.array(AC_rail).mean()
    is_fouled_percentage = np.array(is_fouled).mean()
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
    with open(metadata_dir / "statistics.txt", "w") as f:
        f.write(f"AC rail percentage: {AC_rail_percentage}\n")
        f.write(f"fouled percentage: {is_fouled_percentage}\n")
        f.write(f"water infiltrations percentages: {water_infiltrations_percentages}\n")
        f.write(f"sleeper number distribution: {sleepers_counts_distrib}\n")
        f.write(f"sleeper material distribution: {sleepers_material_counts}\n")

    # plot distributions
    plots_dir = metadata_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # layer sizes
    fig, axs = plt.subplots(ncols=4, sharey=True, tight_layout=True)
    fig.suptitle("layer sizes")
    axs[0].hist(layer_sizes_distrib["fouling"], bins = 100)
    axs[0].set_title("fouling")
    axs[1].hist(layer_sizes_distrib["ballast"], bins = 100)
    axs[1].set_title("ballast")
    axs[2].hist(layer_sizes_distrib["asphalt"], bins = 100)
    axs[2].set_title("asphalt")
    axs[3].hist(layer_sizes_distrib["PSS"], bins = 100)
    axs[3].set_title("PSS")
    fig.savefig(plots_dir / "layer_sizes.png")

    # general water content
    fig, ax = plt.subplots()
    fig.suptitle("General water content distribution")
    ax.hist(general_water_content_distrib, bins = 100)
    fig.savefig(plots_dir / "general_water_content.png")
    
    # layers water ranges
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, tight_layout = True)
    fig.suptitle("water content distributions")
    if is_fouled_percentage > 0.0:
        axs[0].hist2d(layers_water_content_distrib["fouling"][:, 0], layers_water_content_distrib["fouling"][:, 1], bins=20)
    axs[0].set_title("fouling")
    axs[1].hist2d(layers_water_content_distrib["PSS"][:, 0], layers_water_content_distrib["PSS"][:, 1], bins=20)
    axs[1].set_title("PSS")
    axs[2].hist2d(layers_water_content_distrib["subsoil"][:, 0], layers_water_content_distrib["subsoil"][:, 1], bins=20)
    axs[2].set_title("subsoil")
    fig.savefig(plots_dir / "water_content.png")
    plt.close("all")


def create_gprmax_input_files(config: GprMaxConfig):
    """
    Creates the input files needed from gprMax to run the simulations.
    
    Each file also includes geometry save commands to retain the built geometry.

    The intermediate A-scans are set to be written in 'output_dir/tmp/'

    Creates an pickled file containing data about the random values of all the input files created.

    Parameters
    ----------
    config : GprMaxConfig
        gprMax configuration
    """

    metadata = {}
    
    for file_number in tqdm(range(config.n_samples)):
        filename = f"scan_{str(file_number).zfill(4)}"
        file_path = config.input_dir / filename

        with InputFile(file_path.with_suffix(".in"), filename) as f:
            output_dir = config.output_dir / filename
            new_config = config.model_copy(update={"output_dir": output_dir}, deep=True)
            metadata[filename] = f.write_randomized(new_config)
    
    _write_metadata_files(metadata, config.input_dir / "metadata")
    


##############################################
# Create a nostdout context
import contextlib
import sys
class _DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def _nostdout():
    save_stdout = sys.stdout
    sys.stdout = _DummyFile()
    yield
    sys.stdout = save_stdout
###############################################

def run_simulations(input_dir: str | Path, tmp_dir: str | Path, output_dir: str | Path, n_ascans:int, geometry_only: bool):
    """
    Runs the gprMax simulations specified inside the 'input_dir' folder and places its outputs in the 'output_dir' folder.
    Automatically combines the multiple A-scans created for each sample into a single B-scan file.

    Additionally creates gprMax geometry files corresponding to each input file and converts them into numpy format.

    If the 'geometry_only' parameter is set, only creates geometry files, without running the simulations.

    Parameters
    ----------
    input_dir : str | Path
        directory in which to find the fprMax input .in files
    tmp_dir : str | Path
        directory in which to store the intermediate results
    output_dir : str | Path
        directory in which to store the output
    n_ascans : int
        number of A-scans for each generated B-scan
    geometry_only : bool
        if True, gprMax will not rebuild the geometry for every A-scan, but only at the beginning of the B-scan.
    """
    from gprMax import run as gprmax_run
    from tools.outputfiles_merge import merge_files
    input_dir = Path(input_dir)
    tmp_dir = Path(tmp_dir)
    output_dir = Path(output_dir)
    # check if pycuda is installed, otherwise only use cpu
    gpus = None
    try:
        # Check and list any CUDA-Enabled GPUs
        import pycuda.driver as drv
        drv.init()
        if drv.Device.count() == 0:
            raise Exception("No nvidia GPU detected")
        gpus = [0]
        print("NVIDIA GPU detected!")
    except Exception:
        print("No NVIDIA GPU detected, pycuda package or the CUDA toolkit not installed. Falling back to CPU mode.")

    for f in input_dir.glob("*.in"):
        output_files_basename = f.stem
        sim_output_dir = output_dir / output_files_basename
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        # create the temporary snapshot directories
        for i in range(1, n_ascans + 1):
            snapshot_dir = tmp_dir / f"{f.stem}_snaps{i}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

        # run sims
        gprmax_run(str(f), n_ascans, geometry_fixed=False, geometry_only=geometry_only, gpu=gpus)

        # merge output A-scans
        if not geometry_only:
            merged_output_file_name = output_files_basename + "_merged.out"
            merge_files(str(tmp_dir/ output_files_basename), removefiles=True)
            (tmp_dir/merged_output_file_name).rename(sim_output_dir/merged_output_file_name)
        
            # convert the snapshots and save a single npz file, they are in the input folder
            convert_snapshots_to_np(tmp_dir, sim_output_dir / "snapshots", True, (3, 3))
            # delete the empty snapshot directories created by gprMax in the input folder
            dirs = input_dir.glob(f"{f.stem}_snaps*")
            for d in dirs:
                d.rmdir()
        
        # convert output geometry in numpy format and discard initial files
        h5_file_name = output_files_basename + "_geometry.h5"
        convert_geometry_to_np(tmp_dir/h5_file_name, (sim_output_dir/h5_file_name).with_suffix(".npy"), remove_files=False)




if __name__ == "__main__":
    args = _parse_arguments()

    config = vars(args)

    # read the yaml configuration file with the materials:
    with open(args.gprmax_config, "r") as f:
        default_config = safe_load(f)

    default_config.update({k:v for k, v in config.items() if v is not None})
    config = GprMaxConfig(**default_config)

    _resolve_directories(config)

    if config.generate_input:
        create_gprmax_input_files(config)

    run_simulations(config.input_dir, config.tmp_dir, config.output_dir, config.n_ascans, geometry_only=config.geometry_only)
