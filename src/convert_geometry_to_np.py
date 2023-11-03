import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def parse_materials_file(file_path: str | Path):
    with open(file_path, "r") as f:
        materials = f.read().splitlines()

    materials = [l.split() for l in materials if "#material" in l]
    materials = [l[1:-1] for l in materials]
    return materials

def convert_geometry_to_np(filename: str | Path) -> np.ndarray:
    """
    Converts the given geometry and materials files into numpy arrays of shape [3, height, width].

    args:
        - filename (str or Path) : filename of the h5 file containing the geometry. 
            The materials file must be in the same folder and named the same, adding "_materials.txt"

    returns 'ret' (np.ndarray):
     - ret[0] contains the relative permittivity, 
     - ret[1] contains the conductivity,
     - ret[2] contains the relative permeability.
    """
    h5_path = Path(filename).with_suffix(".h5")
    txt_path = h5_path.with_name(h5_path.name + "_materials").with_suffix(".txt")

    materials = parse_materials_file(txt_path)

    h5_file = h5py.File("data/geometry_2D_cylinders_clean.h5")
    data = np.array(h5_file["data"]).squeeze()

    new_data = np.zeros((3, *data.shape), dtype=np.float32)

    for index in range(len(materials)):
        indexes = data == index
        relative_permittivity = materials[index][0]
        conductivity = materials[index][1]
        relative_permeability = materials[index][2]
        new_data[0, indexes] = relative_permittivity
        new_data[1, indexes] = conductivity
        new_data[2, indexes] = relative_permeability

    return new_data
        


if __name__ == "__main__":
    convert_geometry_to_np("data/geometry_2D_cylinders_clean_materials.txt")