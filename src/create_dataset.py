"""
This file is used to create a GPR dataset using gprMax.

First creates some input files based on the provided configuration, then runs gprMax on the input files to get the output.

It also creates geometry files and converts their content into numpy arrays containing the relevant physical values.
"""

import argparse
from pathlib import Path

from gprMax import run as gprmax_run
#TODO: import A-scan joiner

from convert_geometry_to_np import convert_geometry_to_np



def parse_arguments():
    
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("-n", type=int, help="Number of input files to generate/simulations to run.")
    parser.add_argument("--generate_input", action="store_true", 
                        help="If True, generate input files. Otherwise use the files inside '-input_dir'.")
    parser.add_argument("-input_dir", type="str", default="./gprmax_input_files/generated/", 
                        help="Directory to put the generated input files.")
    parser.add_argument("-output_dir", type="str", default="./gprmax_output/",
                        help="Directory to store the generated results.")
    
    # simulation settings
    parser.add_argument("-layer_sizes", nargs=4, type=float default=[0.15, 0.3, 0.55, 0.7],
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
    """
    pass

def run_simulations(input_dir: str | Path, output_dir: str | Path, geometry_only: bool):
    """
    Runs the gprMax simulations specified inside the 'input_dir' folder and places its outputs in the 'output_dir' folder.
    Automatically combines the multiple A-scans created for each sample into a single B-scan file.

    Additionally creates gprMax geometry files corresponding to each input file and converts them into numpy format.

    If the 'geometry_only' parameter is set, only creates geometry files, without running the simulations.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # run sims
    for f in input_dir.glob("*.in"):
        gprmax_run(f, 55, geometry_fixed=True, geometry_only=geometry_only)

    # TODO: combine A-scan files into single B-scan
    pass

    # TODO: convert geometry files into numpy and save them
    pass

if __name__ == "__main__":
    args = parse_arguments()

    if args.generate_input:
        create_gprmax_input_files(args)

    run_simulations(Path(args.input_dir), Path(args.output_dir))
