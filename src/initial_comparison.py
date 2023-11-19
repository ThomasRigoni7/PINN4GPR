import numpy as np
from tools.outputfiles_merge import get_output_data
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def show_bscan(title: str, ax: Axes, data: np.ndarray, dt):
    ax.imshow(data, 
                extent=(0, data.shape[1], data.shape[0] * dt, 0), 
                interpolation='nearest', aspect='auto', cmap='seismic', 
                vmin=-np.amax(np.abs(data)), vmax=np.amax(np.abs(data)))
    ax.set_title("2D boxes")
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time [s]')

def initial_comparison():
    boxes_2D_data, dt0 = get_output_data("gprmax_input_files/initial_comparison/output/2D_boxes_merged.out", 1, "Ez")
    cylinders_2D_data, dt1 = get_output_data("gprmax_input_files/initial_comparison/output/2D_cylinders_pep_merged.out", 1, "Ez")
    cylinders_3D_data, dt2 = get_output_data("gprmax_input_files/initial_comparison/output/3D_cylinders_pep_no_rails_merged.out", 1, "Ez")

    print(boxes_2D_data.shape)
    print(cylinders_2D_data.shape)
    print(cylinders_3D_data.shape)

    fig, axs = plt.subplots(ncols=3)

    time_window_to_remove = 5e-9
    for i, (title, bscan, dt) in enumerate(zip(["2D boxes", "2D cylinders", "3D cylinders"],
                                            [boxes_2D_data, cylinders_2D_data, cylinders_3D_data], [dt0, dt1, dt2])):
        to_delete = int(time_window_to_remove // float(dt))
        bscan[0:to_delete, :] = 0
        print(f"Set the first {to_delete} values to 0!")
        show_bscan(title, axs[i], bscan, dt)



    axs[1].imshow(cylinders_2D_data, 
                extent=[0, cylinders_2D_data.shape[1], cylinders_2D_data.shape[0] * dt1, 0], 
                interpolation='nearest', aspect='auto', cmap='seismic', 
                vmin=-np.amax(np.abs(cylinders_2D_data)), vmax=np.amax(np.abs(cylinders_2D_data)))
    axs[1].set_title("2D cylinders")

    axs[2].imshow(cylinders_3D_data, 
                extent=[0, cylinders_3D_data.shape[1], cylinders_3D_data.shape[0] * dt2, 0], 
                interpolation='nearest', aspect='auto', cmap='seismic', 
                vmin=-np.amax(np.abs(cylinders_3D_data)), vmax=np.amax(np.abs(cylinders_3D_data)))
    axs[2].set_title("3D cylinders")

    plt.show()

if __name__ == "__main__":
    initial_comparison()