"""
This file is used to create a GPR dataset using gprMax.

First creates some input files based on the provided configuration, then runs gprMax on the input files to get the output.

It also creates geometry files and converts their content into numpy arrays containing the relevant physical values.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from yaml import safe_load
import shutil
import numpy as np
import time

from src.dataset_creation.convert_to_np import convert_geometry_to_np, convert_snapshots_to_np
from src.dataset_creation.inputfile import InputFile
from src.dataset_creation.configuration import GprMaxConfig
from src.dataset_creation.statistics import DatasetStats

import contextlib
import sys

@contextlib.contextmanager
def _redirect_stdout(log_file):
    save_stdout = sys.stdout
    save__stdout__ = sys.__stdout__
    sys.stdout = log_file
    sys.__stdout__ = sys.stdout
    yield
    sys.stdout = save_stdout
    sys.__stdout__ = save__stdout__


def _parse_arguments():
    """
    Parses the arguments and returns the derived Namespace.
    """
    
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("config_file", type=str, help="Path to the gprmax yaml config file.")
    parser.add_argument("-ns", "--n_samples", type=int, help="Number of input files to generate/simulations to run.")
    parser.add_argument("-na", "--n_ascans", type=int,
                        help="Number of A-scans that constitute a B-scan")
    parser.add_argument("-i", "--generate_input", action="store_true", 
                        help="If set, generate input files and store them inside `input_dir`.")
    parser.add_argument("-r", "--run_simulations", action="store_true",
                        help="If set, run the simulations inside the input folder.")
    parser.add_argument("--geometry_only", action="store_true",
                        help="If set, only generate the geometries corresponding to the input files, but don't run the simulations.")
    parser.add_argument("--input_dir", type=str,
                        help="Directory to put the generated input files.")
    parser.add_argument("--tmp_dir", type=str,
                        help="Directory to store the gprMax intermediate files.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to store the final results.")
    parser.add_argument("-s", "--seed", type=int,
                        help="The seed used for dataset random generation. The entire generated dataset is deterministic based on this seed.")
    
    args = parser.parse_args()

    # setting geometry only automatically runs the simulations
    if args.geometry_only:
        args.run_simulations = True
    
    if not (args.generate_input or args.run_simulations):
        parser.print_help()
        exit(1)

    return args
    
def _resolve_directories(config: GprMaxConfig):
    """
    Resolves and creates the input, tmp and output directories.
    """
    config.input_dir = config.input_dir.resolve()
    config.tmp_dir = config.tmp_dir.resolve()
    config.output_dir = config.output_dir.resolve()

    # delete the tmp dir to avoid errors in snapshot handling
    tmp_not_empty = config.tmp_dir.exists() and len(list(config.tmp_dir.iterdir())) > 0
    if tmp_not_empty:
        shutil.rmtree(config.tmp_dir)

    config.input_dir.mkdir(exist_ok=True, parents=True)
    config.tmp_dir.mkdir(exist_ok=True, parents=True)
    config.output_dir.mkdir(exist_ok=True, parents=True)


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

    stats = {}
    rng = np.random.default_rng(config.seed)

    print("Generating input files:")    
    for file_number in tqdm(range(config.n_samples)):
        filename = f"scan_{str(file_number).zfill(5)}"
        file_path = config.input_dir / filename

        with InputFile(file_path.with_suffix(".in"), filename) as f:
            output_dir = config.output_dir / filename
            new_config = config.model_copy(update={"output_dir": output_dir}, deep=True)
            stats[filename] = f.write_randomized(new_config, rng.integers(2**31))
    
    stats = DatasetStats(stats)

    stats.write_metadata_files(config.input_dir / "metadata")

    
def check_gpu() -> bool:
    """
    Checks if it is possible to use a GPU in the simulations process.

    Returns
    -------
    bool
        True if it is possible to use a GPU, False otherwise.
    """
    # check if pycuda is installed, otherwise only use cpu
    gpu = False
    try:
        # Check and list any CUDA-Enabled GPUs
        import pycuda.driver as drv
        drv.init()
        if drv.Device.count() == 0:
            raise Exception("No nvidia GPU detected")
        gpu = True
        print("NVIDIA GPU detected!")
    except Exception:
        print("No NVIDIA GPU detected, pycuda package or the CUDA toolkit not installed. Falling back to CPU mode.")

    return gpu

def run_simulations(input_dir: str | Path, tmp_dir: str | Path, output_dir: str | Path, n_ascans:int, geometry_only: bool, gpu: bool):
    """
    Runs the gprMax simulations specified inside the `input_dir` folder and places its outputs in the `output_dir` folder.
    
    Automatically combines the multiple A-scans created for each sample into a single B-scan file.
    Autotatically combines the snapshots created from the scripts into an .npz file.
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
        if set, gprMax will not run the simulations, but only build the file geometries.
    gpu : bool
        if set, try to run gprMax in gpu mode.
    """
    from gprMax import run as gprmax_run
    from tools.outputfiles_merge import merge_files

    print("Running gprMax on the input files:")
    log_file_path = output_dir / "gprmax_output.log"
    print("gprMax output logs are stored in", log_file_path)

    input_dir = Path(input_dir)
    tmp_dir = Path(tmp_dir)
    output_dir = Path(output_dir)

    gpu = [0] if gpu else None

    input_files = list(input_dir.glob("*.in"))
    input_files.sort()

    with open(log_file_path, "w") as logfile:
        for f in tqdm(input_files):
            output_files_basename = f.stem
            sim_output_dir = output_dir / output_files_basename
            sim_output_dir.mkdir(parents=True, exist_ok=True)

            # create the temporary snapshot directories
            for i in range(1, n_ascans + 1):
                snapshot_dir = tmp_dir / f"{f.stem}_snaps{i}"
                snapshot_dir.mkdir(parents=True, exist_ok=True)

            # run sims
            with _redirect_stdout(logfile):
                gprmax_run(str(f), n_ascans, geometry_fixed=False, geometry_only=geometry_only, gpu=gpu)

            # merge output A-scans
            if not geometry_only:
                merged_output_file_name = output_files_basename + "_merged.out"
                if n_ascans > 1:
                    merge_files(str(tmp_dir/ output_files_basename), removefiles=True)
                    (tmp_dir/merged_output_file_name).rename(sim_output_dir/merged_output_file_name)
                elif n_ascans == 1:
                    (tmp_dir/(output_files_basename + ".out")).rename(sim_output_dir/merged_output_file_name)
            
                # convert the snapshots and save a single npz file, they are in the input folder
                convert_snapshots_to_np(tmp_dir, sim_output_dir / "snapshots", True, (3, 3))
                # delete the empty snapshot directories created by gprMax in the input folder
                dirs = input_dir.glob(f"{f.stem}_snaps*")
                for d in dirs:
                    d.rmdir()
            
            # convert output geometry in numpy format and discard initial files
            h5_file_name = output_files_basename + "_geometry.h5"
            convert_geometry_to_np(tmp_dir/h5_file_name, (sim_output_dir/h5_file_name).with_suffix(".npy"), remove_files=True)




if __name__ == "__main__":
    args = _parse_arguments()


    config = vars(args)

    # read the yaml configuration file with the materials:
    with open(args.config_file, "r") as f:
        default_config = safe_load(f)

    default_config.update({k:v for k, v in config.items() if v is not None})
    config = GprMaxConfig(**default_config)

    _resolve_directories(config)

    t = time.time()
    
    if config.generate_input:
        create_gprmax_input_files(config)

    if config.run_simulations:
        gpu_available = check_gpu()
        run_simulations(config.input_dir, config.tmp_dir, config.output_dir, config.n_ascans, 
                        geometry_only=config.geometry_only, 
                        gpu=gpu_available)

    print(f"Done in {time.time() - t} seconds.")
