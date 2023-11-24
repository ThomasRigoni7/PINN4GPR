"""
This module contains utility functions to convert results of the gprMax computation into numpy ndarrays.
"""

import numpy as np
import h5py
from pathlib import Path

def _parse_materials_file(file_path: str | Path) -> list:
    """
    Parses the geometry materials file.
    """
    with open(file_path, "r") as f:
        materials = f.read().splitlines()

    materials = [l.split() for l in materials if "#material" in l]
    materials = [l[1:-1] for l in materials]
    return materials

def convert_geometry_to_np(filename: str | Path, output_file: str | Path | None = None, remove_files: bool = False) -> np.ndarray:
    """
    Converts the given geometry and materials files into numpy arrays of shape [4, height, width].

    Parameters
    ----------
    filename : str | Path
        filename of the h5 file containing the geometry. The materials file must be 
        in the same folder and named the same, adding "_materials.txt"
    output_file : str | Path, default: None
        output file to store the result in .npy format. If 'None', then does not write to disk, but returns the array.
    remove_files : bool, default: False
        if set, deletes the initial h5 and txt files.

    Returns
    -------
    ret : np.ndarray
        - `ret[0]` contains the relative permittivity, 
        - `ret[1]` contains the conductivity,
        - `ret[2]` contains the relative permeability,
        - `ret[3]` contains the magnetic loss.
    """
    h5_path = Path(filename).with_suffix(".h5")
    txt_path = h5_path.with_name(h5_path.with_suffix("").name + "_materials").with_suffix(".txt")

    materials = _parse_materials_file(txt_path)

    h5_file = h5py.File(h5_path)
    data = np.array(h5_file["data"]).squeeze()

    new_data = np.zeros((4, *data.shape), dtype=np.float32)

    for index in range(len(materials)):
        indexes = data == index
        relative_permittivity = materials[index][0]
        conductivity = materials[index][1]
        relative_permeability = materials[index][2]
        magnetic_loss = materials[index][3]
        new_data[0, indexes] = relative_permittivity
        new_data[1, indexes] = conductivity
        new_data[2, indexes] = relative_permeability
        new_data[3, indexes] = magnetic_loss

    if remove_files:
        h5_path.unlink()
        txt_path.unlink()

    if output_file is None:
        return new_data
    else:
        np.save(output_file, new_data)
        


if __name__ == "__main__":
    convert_geometry_to_np("data/geometry_2D_cylinders_clean_materials.txt")