"""
This module contains utility functions to convert results of the gprMax computation into numpy ndarrays.
"""

from attr import field
import numpy as np
import h5py
from pathlib import Path

import vtkmodules

from gprmax_repo import gprMax

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
        
def extract_snapshot_fields_numpy(snapshot_path: str | Path):
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(snapshot_path)
    reader.Update()
    data = reader.GetOutput()
    cell_data = data.GetCellData()
    vtk_e_field : vtk.vtkTypeFloat32Array
    vtk_e_field = cell_data.GetArray("E-field")
    e_field = vtk_to_numpy(vtk_e_field).copy()
    vtk_h_field = cell_data.GetArray("H-field")
    h_field = vtk_to_numpy(vtk_h_field).copy()
    dims = data.GetDimensions()
    # print(e_field.shape)
    # print(h_field.shape)
    # print(dims)

    # E-field has only z component,  H-field only x,y
    assert np.allclose(e_field[:, 0], np.zeros_like(e_field[:, 0]))
    assert np.allclose(e_field[:, 1], np.zeros_like(e_field[:, 1]))
    assert np.allclose(h_field[:, 2], np.zeros_like(h_field[:, 2]))

    # possibly an error in gprMax with the specification of the dimentions
    dims = dims[0] - 1, dims[1] - 1, 3
    e_field = e_field.reshape(dims[1], dims[0], 3)
    e_field = np.flipud(e_field)
    h_field = h_field.reshape(dims[1], dims[0], 3)
    h_field = np.flipud(h_field)

    return e_field, h_field

def convert_snapshots_to_np(output_folder : str | Path, remove_files: bool = False):
    """
    Converts snapshots related to a full B-scan into numpy arrays.

    Creates a single npz file containing all the arrays. Each array inside the npz 
    file is of shape [n_snapshots, height, width]

    Parameters
    ----------
    output_folder : str | Path
        the output folder of the B-scan.
    remove_files : bool, default: False
        if set, deletes the original .vti files.
    """
    output_folder = Path(output_folder)
    folders = output_folder.glob("scan_*_snaps*")
    folders = [f for f in folders]
    print(folders)
    
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # convert_geometry_to_np("data/geometry_2D_cylinders_clean_materials.txt")
    # e_field, h_field = extract_snapshot_fields_numpy("gprmax_output_files/scan_0000/scan_0000_snaps1/snap_1.1e-08.vti")
    # print(e_field.shape)
    # plt.imshow(e_field[:, :, 2], cmap='seismic',
    #            vmin=-np.absolute(e_field).max(), vmax=np.absolute(e_field).max())
    # plt.show()
    # plt.imshow(h_field[:, :, 0], cmap='seismic',
    #            vmin=-np.absolute(h_field).max(), vmax=np.absolute(h_field).max())
    # plt.show()
    convert_snapshots_to_np("gprmax_output_files/scan_0000")
