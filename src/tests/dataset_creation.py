from yaml import safe_load
from difflib import unified_diff
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tools.plot_Bscan import get_output_data
from tqdm import tqdm
import cv2

from src.dataset_creation.create_dataset import create_gprmax_input_files, run_simulations, _resolve_directories
from src.dataset_creation.inputfile import InputFile
from src.dataset_creation.configuration import GprMaxConfig


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

    config.n_samples = 5

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

    
def test_bscan_pml_bug(dataset_location: str | Path):
    dataset_location = Path(dataset_location)
    output_folder = dataset_location / "gprmax_output_files"
    print("Loading dataset...")
    bugs_count = 0
    vals = []
    for i in tqdm(range(900)):
        bscan_path = output_folder / f"scan_{str(i).zfill(5)}" / f"scan_{str(i).zfill(5)}_merged.out"
        bscan, dt = get_output_data(bscan_path, 1, "Ez")
        bscan = cv2.resize(bscan, (192, 224))
        val = np.abs(bscan).sum()
        vals.append(val)
        if val > 1.4e6:
            bugs_count += 1
    
    counts, edges, bars = plt.hist(vals)
    plt.bar_label(bars)
    plt.xlabel("sum of abs value")
    plt.ylabel("count")
    plt.show()

    print("BUGS COUNT:", bugs_count)
    print("total bscans:", 900)

if __name__ == "__main__":
    # test_inputfile_deterministic()
    # test_inputfile_construction()
    test_bscan_pml_bug("dataset_bscan")