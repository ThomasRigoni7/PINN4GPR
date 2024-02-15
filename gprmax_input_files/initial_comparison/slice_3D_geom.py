import h5py
import numpy as np

h5_file = h5py.File("gprmax_input_files/initial_comparison/3D_spheres_geom.h5")

print(h5_file.keys())

print("ID:", h5_file["ID"])
print("data:", h5_file["data"])
print("rigidE:", h5_file["rigidE"])
print("rigidH:", h5_file["rigidH"])

data = np.array(h5_file["data"])
slice = data[:, :, 200]
slice = slice[:, :, None]
attrs = [i for i in h5_file.attrs.items()]
print("old attributes:", attrs)
attrs[0] = ("Title", "2D_spheres")
print("new attributes:", attrs)


with h5py.File("gprmax_input_files/initial_comparison/2D_spheres_geom.h5", "w") as f:
    f.create_dataset("data", data=slice, dtype="i2")
    for a_name, a_data in attrs:
        f.attrs.create(a_name, a_data)
