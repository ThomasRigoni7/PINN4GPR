"""
This file is used to create a GPR dataset using gprMax.

First creates some input files based on the provided configuration, then runs gprMax on the input files to get the output.

It also creates geometry files and converts their content into numpy arrays containing the relevant physical values.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from yaml import safe_load

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
    

def create_gprmax_input_files(config: GprMaxConfig):
    """
    Creates the input files needed from gprMax to run the simulations.
    
    Each file also includes geometry save commands to retain the built geometry.

    The intermediate A-scans are set to be written in 'output_dir/tmp/'

    Parameters
    ----------
    config : GprMaxConfig
        gprMax configuration
    """
    
    for file_number in tqdm(range(config.n_samples)):
        filename = f"scan_{str(file_number).zfill(4)}"
        file_path = config.input_dir / filename

        with InputFile(file_path.with_suffix(".in"), filename) as f:
            output_dir = config.output_dir / filename
            new_config = config.model_copy(update={"output_dir": output_dir}, deep=True)
            f.write_randomized(new_config)

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
        # run sims
        # with _nostdout():
        output_files_basename = f.stem
        sim_output_dir = output_dir / output_files_basename
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        # create the temporary snapshot directories
        for i in range(1, config.n_ascans + 1):
            snapshot_dir = tmp_dir / f"{f.stem}_snaps{i}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

        gprmax_run(str(f), n_ascans, geometry_fixed=False, geometry_only=geometry_only, gpu=gpus)

        # merge output A-scans
        if not geometry_only:
            merged_output_file_name = output_files_basename + "_merged.out"
            merge_files(str(tmp_dir/ output_files_basename), removefiles=True)
            (tmp_dir/merged_output_file_name).rename(sim_output_dir/merged_output_file_name)
        
        # convert output geometry in numpy format and discard initial files
        h5_file_name = output_files_basename + "_geometry.h5"
        convert_geometry_to_np(tmp_dir/h5_file_name, (sim_output_dir/h5_file_name).with_suffix(".npy"), remove_files=True)

        # convert the snapshots and save a single npz file, they are in the input folder
        convert_snapshots_to_np(tmp_dir, sim_output_dir / "snapshots", True)
        # delete the empty snapshot directories created by gprMax in the input folder
        dirs = input_dir.glob(f"{f.stem}_snaps*")
        for d in dirs:
            d.rmdir()



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

    #run_simulations(config.input_dir, config.tmp_dir, config.output_dir, config.n_ascans, geometry_only=False)
