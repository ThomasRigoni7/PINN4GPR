"""
This module contains utility functions to convert results of the gprMax computation into numpy ndarrays.
"""

import numpy as np
from skimage.measure import block_reduce
import h5py
from pathlib import Path

def _update_debye_er(materials_list: list[str], debye_materials: list[str], frequency: float=1e9):
    """
    Calculates the relative permittivity of the Debye materials at the specified frequency, 
    given the list of materials and debye declarations.

    Most of this code has been adapted from gprMax source code.

    Returns
    -------
    list 
        the list of materials, with updated relative permittivity.
    """
    from gprMax.materials import Material
    for dm in debye_materials:
        name = dm[-1]
        mat = [m for m in materials_list if name in m]
        if len(mat) == 0:
            raise ValueError(f"Error: trying to calculate Debye epsilon r for material '{name}', but material declaration not found!")
        mat = mat[0]

        # setup material
        m = Material(0, name)
        m.er = float(mat[0])
        m.se = float(mat[1])
        m.mr = float(mat[2])
        m.sm = float(mat[3])
        poles = int(dm[0])
        m.type = 'debye'
        m.poles = poles
        m.averagable = False
        for pole in range(1, 2 * poles, 2):
            if float(dm[pole]) > 0:
                m.deltaer.append(float(dm[pole]))
                m.tau.append(float(dm[pole + 1]))
        
        # calculate er
        er = m.calculate_er(frequency)
        index = materials_list.index(mat)
        materials_list[index][0] = er.real

    return materials_list


def _parse_materials_file(file_path: str | Path) -> list:
    """
    Parses the geometry materials file.
    """
    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    materials = [l.split() for l in lines if "#material" in l]
    materials = [l[1:] for l in materials]

    debye_materials = [l.split() for l in lines if "#add_dispersion_debye" in l]
    debye_materials = [l[1:] for l in debye_materials]

    materials = _update_debye_er(materials, debye_materials)
    materials = [l[:-1] for l in materials]

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
    ret : np.ndarray of shape [4, height, width]
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
    
    # fix axis order and orientation:
    new_data = np.flip(new_data.transpose(0, 2, 1), 1)

    if remove_files:
        h5_path.unlink()
        txt_path.unlink()

    if output_file is None:
        return new_data
    else:
        np.save(output_file, new_data)
        
def extract_snapshot_fields_numpy(snapshot_path: str | Path):
    """
    Extract the electric and magnetic fields from a snapshot `.vti` file.

    Parameters
    ----------
    snapshot_path : str | Path
        path to the snapshot file generated from gprMax

    Returns
    -------
    tuple[np.ndarray, np.ndarray] of shape [height, width, 3]
        Electric and magnetic fields in the x, y, z directions
    """
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(snapshot_path))
    reader.Update()
    data = reader.GetOutput()
    cell_data = data.GetCellData()
    vtk_e_field : vtk.vtkTypeFloat32Array
    vtk_e_field = cell_data.GetArray("E-field")
    e_field = vtk_to_numpy(vtk_e_field).copy()
    vtk_h_field = cell_data.GetArray("H-field")
    h_field = vtk_to_numpy(vtk_h_field).copy()
    dims = data.GetDimensions()

    # Possibly an error in gprMax with the specification of the array sizes
    dims = dims[0] - 1, dims[1] - 1, 3
    e_field = e_field.reshape(dims[1], dims[0], dims[2])
    e_field = np.flipud(e_field)
    h_field = h_field.reshape(dims[1], dims[0], dims[2])
    h_field = np.flipud(h_field)

    return e_field, h_field

def convert_snapshots_to_np(snapshot_folder : str | Path, 
                            output_file: str | Path, 
                            remove_files: bool = False, 
                            pool_window: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """
    Converts snapshots related to a full B-scan into numpy arrays.

    Creates a single `.npz` file containing all the arrays. 
    
    Each array inside the npz file is of shape:
     - [n_snapshots, height, width] for E-fields (z component of the field)
     - [n_snapshots, 2, height, width] for H-fields (x, y components)

    Parameters
    ----------
    snapshot_folder : str | Path
        path to the folder containing all the snapshots of a B-scan.
    output_file : str | Path
        path to the output to store the results in `.npz` format
    remove_files : bool, default: False
        if set, deletes the original `.vti` files.
    pool_window : tuple[int, int] | None, default: None
        if set, applies average pooling of the specified size to the resulting snapshot.

    Returns
    -------
    dict[str, np.ndarray]
        the in-memory dictionary that has been saved to disk
    """
    snapshot_folder = Path(snapshot_folder)
    folders = snapshot_folder.glob("scan_*_snaps*")
    folders = [f for f in folders]

    def extract_time_from_snapshot_path(snapshot_file_path : Path):
        time = float(snapshot_file_path.stem.lstrip("snap_"))
        return time

    full_data = {}
    for folder in folders:
        a_scan_number = int(folder.stem[15:]) - 1
        snapshot_files = [f for f in folder.glob("snap_*.vti")]
        snapshot_files.sort(key=extract_time_from_snapshot_path)
        times = [extract_time_from_snapshot_path(f) for f in snapshot_files]
        times.sort()

        a_scan_data_e_field = []
        a_scan_data_h_field = []
        for file in snapshot_files:
            e_field, h_field = extract_snapshot_fields_numpy(file)
            e_field = e_field[:, :, 2]
            h_field = h_field[:, :, 0:2]
            h_field = h_field.transpose(2, 0, 1)
            a_scan_data_e_field.append(e_field)
            a_scan_data_h_field.append(h_field)
            
            if remove_files:
                file.unlink()

        a_scan_data_e_field = np.asarray(a_scan_data_e_field)
        a_scan_data_h_field = np.asarray(a_scan_data_h_field)
        
        # apply pooling
        if pool_window is not None:
            a_scan_data_e_field = block_reduce(a_scan_data_e_field, block_size=(1, pool_window[0], pool_window[1]), func=np.mean)
            a_scan_data_h_field = block_reduce(a_scan_data_h_field, block_size=(1, 1, pool_window[0], pool_window[1]), func=np.mean)
        full_data[f"{str(a_scan_number).zfill(4)}_E"] = a_scan_data_e_field
        full_data[f"{str(a_scan_number).zfill(4)}_H"] = a_scan_data_h_field
        full_data[f"{str(a_scan_number).zfill(4)}_times"] = times

        if remove_files:
            folder.rmdir()

    np.savez(output_file, **full_data)

    return full_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # convert_geometry_to_np("data/geometry_2D_cylinders_clean_materials.txt")
    data = convert_snapshots_to_np("gprmax_output_files_old/scan_0000", "scan_0000_snapshots")
    pooled_data = convert_snapshots_to_np("gprmax_output_files_old/scan_0000", "scan_0000_snapshots_pooled", pool_window=(3, 3))
    print(data.keys())
    from ..visualization.misc import save_field_animation
    save_field_animation(data["0000_E"], "gprmax_output_files_old/scan_0000/snapshots0.mp4")
    save_field_animation(pooled_data["0000_E"], "gprmax_output_files_old/scan_0000/snapshots0_pooled.mp4")
