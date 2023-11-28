import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def plot_bscan(title: str, ax: Axes, bscan: np.ndarray, dt: float):
    """
    Plots a B-scan in the specitied subplot.

    Parameters
    ----------
    title : str
        the title of the B-scan
    ax : Axes
        the subplot in which to plot
    bscan : np.ndarray
        the B-scan to plot
    dt : float
        the temporal resolution of the B-scan.
    """
    ax.imshow(bscan, 
                extent=(0, bscan.shape[1], bscan.shape[0] * dt, 0), 
                interpolation='nearest', aspect='auto', cmap='seismic', 
                vmin=-np.absolute(bscan).max(), vmax=np.absolute(bscan).max())
    #            vmin=-10, vmax=10)
    ax.set_title(title)
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time [s]')

def _plot_corrected_diff(title: str, ax: Axes, data1, data2, dt):
    ascan1 = data1[:, 30]
    ascan2 = data2[:, 30]

    from scipy import fftpack
    A = fftpack.fft(ascan1)
    B = fftpack.fft(ascan2)
    Br = -B.conjugate()
    offset = np.argmax(np.abs(fftpack.ifft(Br * A)))

    ascan2 = np.roll(ascan2, offset)
    adj_diff = ascan1 - ascan2
    adj_diff[0:offset] = 0

    plot_bscan(title, ax, adj_diff, dt)

def plot_ascans(bscan1: np.ndarray, bscan2: np.ndarray, labels:tuple[str, str], colors: tuple[str, str] = ["r", "b"]) -> np.ndarray:
    """
    Plots and shows the central A-scans related to the provided B-scans, together with their difference.

    Parameters
    ----------
    bscan1 : np.ndarray
        the first B-scan.
    bscan2 : np.ndarray
        the second B-scan, its shape must be the same as `bscan1.shape`
    labels : tuple[str, str]
        the labels to put in the plot legend
    colors : tuple[str, str], default: ["r", "b"]
        the colors to use for the plot

    Returns
    -------
    np.ndarray
        a copy of the second B-scan, temporally shifted to match the first one.
    """
    ascan1 = bscan1[:, bscan1.shape[1]//2]
    ascan2 = bscan2[:, bscan2.shape[1]//2]

    from scipy import fftpack
    A = fftpack.fft(ascan1)
    B = fftpack.fft(ascan2)
    Br = -B.conjugate()
    offset = np.argmax(np.abs(fftpack.ifft(Br * A)))

    diff = ascan1 - ascan2


    adj_std_3D_Ascan = np.roll(ascan2, offset)
    adj_diff = ascan1 - adj_std_3D_Ascan
    adj_diff[0:offset] = 0

    axs : list[Axes]
    fig, axs = plt.subplots(nrows=2)

    axs[0].plot(ascan1, colors[0], label=labels[0] + " A-scan")
    axs[0].plot(ascan2, colors[1], label=labels[1] + " A-scan")
    axs[0].legend()
    axs[0].set_ylim(-10, 10)
    axs[1].plot(diff, "g", label=f"{labels[0]}-{labels[1]} difference")
    axs[1].plot(adj_diff, "m", label=f"{labels[0]}-{labels[1]} adjusted difference")
    axs[1].legend()
    axs[1].set_ylim(-10, 10)
    plt.show()

    data2_rolled = np.roll(bscan2, offset, 0)
    data2_rolled[0:offset, :] = 0
    return data2_rolled

def initial_comparison():
    """
    Perform the initial comparison between different 2D and 3D geometries.

    Compares the following models:
     - 2D box
     - 2D cylindrical
     - 3D cylindrical
     - 3D cylindrical with rails
    """
    import cv2
    from tools.outputfiles_merge import get_output_data
    boxes_2D_data, dt0 = get_output_data("gprmax_input_files/initial_comparison/output/2D_boxes_merged.out", 1, "Ez")
    cylinders_2D_data, dt1 = get_output_data("gprmax_input_files/initial_comparison/output/2D_cylinders_pep_merged.out", 1, "Ez")
    cylinders_3D_data, dt2 = get_output_data("gprmax_input_files/initial_comparison/output/3D_cylinders_pep_no_rails_merged.out", 1, "Ez")
    cylinders_3D_rail_data, dt3 = get_output_data("gprmax_input_files/initial_comparison/output/3D_cylinders_pep_rails_merged.out", 1, "Ez")

    assert dt2 == dt3, "Error: time discretization is different between 3D models."

    data = [boxes_2D_data, cylinders_2D_data, cylinders_3D_data, cylinders_3D_rail_data]
    dts = [dt0, dt1, dt2, dt3]

    time_window_to_remove = 5e-9
    for bscans, dt in zip(data, dts):
        to_delete = int(time_window_to_remove // float(dt))
        bscans[0:to_delete, :] = 0
        print(f"Set the first {to_delete} values to 0!")

    resized_cylinders_2D_data = cv2.resize(cylinders_2D_data, (cylinders_3D_data.shape[1], cylinders_3D_data.shape[0]))
    resized_box = cv2.resize(boxes_2D_data, (cylinders_3D_data.shape[1], cylinders_3D_data.shape[0]))

    std_box = (resized_box - resized_box.mean()) / resized_box.std()
    std_2D = (resized_cylinders_2D_data - resized_cylinders_2D_data.mean()) / resized_cylinders_2D_data.std()
    std_3D = (cylinders_3D_data - cylinders_3D_data.mean()) / cylinders_3D_data.std()
    std_3D_rail = (cylinders_3D_rail_data - cylinders_3D_rail_data.mean()) / cylinders_3D_rail_data.std()

    plot_ascans(cylinders_3D_data, cylinders_3D_rail_data, ["3Dc", "3Dr"], ["", "g"])
    fig, axs = plt.subplots(ncols=2)
    plot_bscan("3D", axs[0], cylinders_3D_data, dt2)
    plot_bscan("3D rail", axs[1], cylinders_3D_rail_data, dt2)
    plt.show()

    adj_box = plot_ascans(std_2D, std_box, ["2Dc", "2Db"], ["r", "k"])
    adj_3D = plot_ascans(std_2D, std_3D, ["2Dc", "3Dc"], ["r", "g"])
    adj_3D_rail = plot_ascans(std_2D, std_3D_rail, ["2Dc", "3Dr"], ["r", "b"])
    plot_ascans(std_3D, std_3D_rail, ["3Dc", "3Dr"], ["g", "b"])

    fig, axs = plt.subplots(ncols=4, nrows=2)

    titles = [["2D boxes", "2D cylinders", "3D cylinders", "3D cylinders rail"],
              ["2Dc - 2Db", "2Dc - 3Dc", "2Dc - 3Dr", "3Dc - 3Dr"]]
    data = [[std_box, std_2D, std_3D, std_3D_rail],
            [std_2D - adj_box, std_2D - adj_3D, std_2D - adj_3D_rail, adj_3D - adj_3D_rail]]

    for i, (titles, bscans) in enumerate(zip(titles, data)):
        for j, (title, bscan) in enumerate(zip(titles, bscans)):
            plot_bscan(title, axs[i][j], bscan, dt2)

    plt.show()

if __name__ == "__main__":
    initial_comparison()