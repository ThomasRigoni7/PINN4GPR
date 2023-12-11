from yaml import safe_load
from difflib import unified_diff
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from ..dataset_creation.create_dataset import create_gprmax_input_files, run_simulations, _resolve_directories
from ..dataset_creation.inputfile import InputFile
from ..dataset_creation.configuration import GprMaxConfig


def test_inputfile_deterministic(delete_files:bool = True):

    test_path_1 = Path("test_0001.in")
    test_path_2 = Path("test_0002.in")

    with open("gprmax_config.yaml", "r") as f:
        default_config = safe_load(f)
    config = GprMaxConfig(**default_config)
    
    with InputFile(test_path_1, "test") as file1, InputFile(test_path_2, "test") as file2:

        file1.write_randomized(config, 42)
        file2.write_randomized(config, 42)

    with open(test_path_1, "r") as f1, open(test_path_2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff = [l for l in unified_diff(lines1, lines2, n=0)]

    if len(diff) == 0:
        print("Inputfile deterministic!")
    else:
        print("Inputfile not deterministic!")
        print("DIFF:")
        for l in diff:
            print(l, end="")
    
    if delete_files:
        test_path_1.unlink()
        test_path_2.unlink()


def test_inputfile_construction():
    with open("gprmax_config.yaml", "r") as f:
        default_config = safe_load(f)
    config = GprMaxConfig(**default_config)
    config.input_dir = Path("test_inputfiles/")
    config.output_dir = Path("test_outputfiles/")
    config.snapshot_times = []

    config.n_samples = 20

    _resolve_directories(config)

    create_gprmax_input_files(config)

    run_simulations(config.input_dir, config.tmp_dir, config.output_dir, 1, True)

    geometries = [f for f in Path("test_outputfiles/").glob("**/*.npy")]
    geometries.sort()

    for f in geometries:
        data = np.load(f)
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(data[0])
        axs[1].imshow(data[1])
        fig.suptitle(f.stem)
        plt.show()

    


if __name__ == "__main__":
    test_inputfile_deterministic()
    test_inputfile_construction()