"""
This file is used to create a GPR dataset using gprMax.

First creates some input files based on the provided configuration, then runs gprMax on the input files to get the output.

It also creates geometry files and converts their content into numpy arrays containing the relevant physical values.
"""

import argparse
from pathlib import Path

from gprMax import run as gprmax_run
from tools.outputfiles_merge import merge_files

from convert_geometry_to_np import convert_geometry_to_np
import numpy as np



def parse_arguments():
    
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("-n", type=int, help="Number of input files to generate/simulations to run.")
    parser.add_argument("-n_ascans", type=int, default=55, 
                        help="Number of A-scans that constitute a B-scan, default=55")
    parser.add_argument("--generate_input", action="store_true", 
                        help="If True, generate input files. Otherwise use the files inside '-input_dir'.")
    parser.add_argument("-input_dir", type=str, default="./gprmax_input_files/generated/", 
                        help="Directory to put the generated input files.")
    parser.add_argument("-output_dir", type=str, default="./gprmax_output/",
                        help="Directory to store the generated results.")
    parser.add_argument("-geometry_only", action="store_true",
                        help="If set, only generate the geometries corresponding to the input files, but don't run the simulations.")
    
    # simulation settings
    parser.add_argument("-layer_sizes", nargs=4, type=float, default=[0.15, 0.3, 0.55, 0.7],
                        help="Sizes of the gravel/asphalt/pss/ballast layers. Interpreted as cumulative height.")
    parser.add_argument("-sleepers_separation", type=float, default=0.65,
                        help="Separation between the sleepers in meters.")
    parser.add_argument("-sleepers_material", type=str, choices=["all", "steel", "concrete", "wood"], default="all")
    parser.add_argument("-max_fouling_level", type=float, default=0.15,
                        help="Maximum ballast fouling height in meters, measured from the pss layer interface.")
    parser.add_argument("-max_fouling_water", type=float, default=0.15,
                        help="Maximum percentage of water in fouling material between ballast stones. Default 0.15 means 15%.")
    parser.add_argument("-max_pss_water", type=float, default=0.15,
                        help="Maximum percentage of water in the pss material. Default 0.15 means 15%.")
    
    args = parser.parse_args()
    return args
    
def create_gprmax_input_files(args):
    """
    Creates the input files needed from gprMax to run the simulations.
    
    Each file also includes geometry save commands to retain the built geometry.

    The intermediate A-scans are set to be written in 'output_dir/tmp/'
    """
    
    # various paths:
    # input_files path
    # output_files path
    # tmp_path is out_files/tmp 
    # output_dir and geometry files path to set inside the input file
    
    pass



def run_simulations(input_dir: str | Path, output_dir: str | Path, n_ascans:int, geometry_only: bool):
    """
    Runs the gprMax simulations specified inside the 'input_dir' folder and places its outputs in the 'output_dir' folder.
    Automatically combines the multiple A-scans created for each sample into a single B-scan file.

    Additionally creates gprMax geometry files corresponding to each input file and converts them into numpy format.

    If the 'geometry_only' parameter is set, only creates geometry files, without running the simulations.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    tmp_dir = output_dir/"tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    for f in input_dir.glob("*.in"):
        # run sims
        gprmax_run(str(f), n_ascans, geometry_fixed=True, geometry_only=geometry_only)

        # merge output A-scans
        if not geometry_only:
            output_files_basename = f.with_suffix("").name
            merged_output_file_name = output_files_basename + "_merged.out"
            merge_files(str(tmp_dir/ output_files_basename), removefiles=True)
            (tmp_dir/merged_output_file_name).rename(output_dir/merged_output_file_name)
        
        # convert output geometry in numpy format and discard initial files
        h5_file_name = output_files_basename + "_geometry.h5"
        convert_geometry_to_np(tmp_dir/h5_file_name, (output_dir/h5_file_name).with_suffix(".npy"), remove_files=True)

if __name__ == "__main__":
    args = parse_arguments()

    if args.generate_input:
        create_gprmax_input_files(args)

    run_simulations(Path(args.input_dir), Path(args.output_dir), args.n_ascans, geometry_only=False)
