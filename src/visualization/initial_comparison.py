import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import mean_squared_error
from scipy import fftpack

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 9),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

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
    #            vmin=-np.absolute(bscan).max(), vmax=np.absolute(bscan).max())
                vmin=-10, vmax=10)
    ax.set_title(title)
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time [s]')

def get_bscan_mse(bscan1: np.ndarray, bscan2: np.ndarray):
    """
    Computes the MSE between flattened versions of the B-scans

    Parameters
    ----------
    bscan1 : np.ndarray
        the first B-scan
    bscan2 : np.ndarray
        the second B-scan

    Returns
    -------
    float
        value of the mean squared error
    """
    b1 = bscan1.flatten()
    b2 = bscan2.flatten()
    return mean_squared_error(b1, b2)

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

    # calculate the necessary offset to make the scans match
    A = fftpack.fft(ascan1)
    B = fftpack.fft(ascan2)
    Br = -B.conjugate()
    offset = np.argmax(np.abs(fftpack.ifft(Br * A)))

    
    adj_ascan2 = np.roll(ascan2, offset)
    adj_diff = ascan1 - adj_ascan2
    adj_diff[0:offset] = 0

    mse = mean_squared_error(ascan1, adj_ascan2)
    print(f"A-scan MSE {labels[0]}-{labels[1]}:", mse)

    axs : list[Axes]
    fig, axs = plt.subplots(nrows=2)

    axs[0].plot(ascan1, colors[0], label=labels[0] + " A-scan")
    axs[0].plot(adj_ascan2, colors[1], label="adj." + labels[1] + " A-scan")
    axs[0].legend()
    axs[0].set_ylim(-10, 10)
    axs[0].set_xlabel("time step")
    axs[0].set_ylabel("std. E field value")
    # axs[1].plot(diff, "g", label=f"{labels[0]}-{labels[1]} difference")
    axs[1].plot(adj_diff, "m", label=f"{labels[0]}-{labels[1]} adjusted difference")
    axs[1].legend()
    axs[1].set_ylim(-10, 10)
    axs[1].set_xlabel("time step")
    axs[1].set_ylabel("std. E field value")
    plt.show()

    bscan2_rolled = np.roll(bscan2, offset, 0)
    bscan2_rolled[0:offset, :] = 0
    return bscan2_rolled

def initial_comparison():
    """
    Performs the initial comparison between different 2D and 3D geometries.

    Compares the following models:
     - 2D box
     - 2D cylindrical 
     - 3D cylindrical
     - 3D cylindrical with rails
     - 3D spheres with rails
     - 2D spheres cut from the 3D geometry

    Where all the models use a Peplinski soil model, except for the 2D box.
    """
    import cv2
    from tools.outputfiles_merge import get_output_data
    boxes_2D_data, dt0 = get_output_data("gprmax_input_files/initial_comparison/output/2D_boxes.out", 1, "Ez")
    circles_2D_data, dt1 = get_output_data("gprmax_input_files/initial_comparison/output/2D_cylinders.out", 1, "Ez")

    cylinders_3D_data, dt2 = get_output_data("gprmax_input_files/initial_comparison/output/3D_cylinders.out", 1, "Ez")
    cylinders_3D_rail_data, dt3 = get_output_data("gprmax_input_files/initial_comparison/output/3D_cylinders_rails.out", 1, "Ez")

    spheres_2D_data, dt4 = get_output_data("gprmax_input_files/initial_comparison/output/2D_spheres.out", 1, "Ez")
    spheres_3D_data, dt5 = get_output_data("gprmax_input_files/initial_comparison/output/3D_spheres.out", 1, "Ez")

    assert dt2 == dt3 == dt5, "Error: time discretization is different between 3D models."

    data = [boxes_2D_data, circles_2D_data, cylinders_3D_data, cylinders_3D_rail_data, spheres_2D_data, spheres_3D_data]
    dts = [dt0, dt1, dt2, dt3, dt4, dt5]

    time_window_to_remove = 5e-9
    for bscans, dt in zip(data, dts):
        to_delete = int(time_window_to_remove // float(dt))
        bscans[0:to_delete, :] = 0
        print(f"Set the first {to_delete} values to 0!")

    # Since the time discretizations are different, we need to resize the 2D B-scans to the 3D size
    (resized_circles_2D_data) = cv2.resize(circles_2D_data, (cylinders_3D_data.shape[1], cylinders_3D_data.shape[0]))
    resized_box = cv2.resize(boxes_2D_data, (cylinders_3D_data.shape[1], cylinders_3D_data.shape[0]))
    resized_spheres_2D = cv2.resize(spheres_2D_data, (cylinders_3D_data.shape[1], cylinders_3D_data.shape[0]))

    # calculate scaling factors
    stds_2D = [resized_box.std(), (resized_circles_2D_data).std(), resized_spheres_2D.std()]
    std_factor_2D = np.asarray(stds_2D).mean()
    stds_3D = [cylinders_3D_data.std(), cylinders_3D_rail_data.std(), spheres_3D_data.std()]
    std_factor_3D = np.asarray(stds_3D).mean()

    print("stds_2D:", stds_2D)
    print("STD resize factor 2D:", std_factor_2D)
    print("stds_3D:", stds_3D)
    print("STD resize factor 3D:", std_factor_3D)

    # rescale the values
    std_2D_box = (resized_box) / std_factor_2D
    std_2D_circles = ((resized_circles_2D_data)) / std_factor_2D
    std_3D_cylinders = (cylinders_3D_data) / std_factor_3D
    std_3D_cylinders_rail = (cylinders_3D_rail_data) / std_factor_3D
    std_3D_spheres = spheres_3D_data / std_factor_3D
    std_2D_spheres = resized_spheres_2D / std_factor_2D

    adj_box = plot_ascans(std_2D_circles, std_2D_box, ["2Dc", "2Db"], ["r", "k"])
    adj_3D_spheres = plot_ascans(std_2D_circles, std_3D_spheres, ("2Dc", "3Ds"), ("r", "b"))

    adj_3D = plot_ascans(std_2D_circles, std_3D_cylinders, ["2Dc", "3Dc"], ["r", "g"])
    adj_3D_rail = plot_ascans(adj_3D, std_3D_cylinders_rail, ["3Dc", "3Dr"], ["g", "b"])
    
    fig, axs = plt.subplots(ncols=3)
    plot_bscan("3D cylinders", axs[0], std_3D_cylinders, dt2)
    plot_bscan("3D cylinders rail", axs[1], std_3D_cylinders_rail, dt3)
    plot_bscan("Difference", axs[2], std_3D_cylinders - std_3D_cylinders_rail, dt3)
    plt.show()

    plot_ascans(std_3D_cylinders_rail, std_3D_spheres, ("3Dr", "3Ds"), ("g", "b"))
    plot_ascans(std_2D_spheres, std_3D_spheres, ("2Ds", "3Ds"), ("grey", "b"))
    adj_2D_spheres = plot_ascans(std_2D_circles, std_2D_spheres, ("2Dc", "2Ds"), ("r", "grey"))
    
    print("B-scans MSE:")
    print("2Db-2Dc:", get_bscan_mse(adj_box, std_2D_circles))
    print("2Db-3Dc:", get_bscan_mse(adj_box, adj_3D))
    print("2Db-3Dr:", get_bscan_mse(adj_box, adj_3D_rail))
    print("2Db-3Ds:", get_bscan_mse(adj_box, adj_3D_spheres))
    print("2Db-2Ds:", get_bscan_mse(adj_box, adj_2D_spheres))
    print()

    print("2Dc-3Dc:", get_bscan_mse(std_2D_circles, adj_3D))
    print("2Dc-3Dr:", get_bscan_mse(std_2D_circles, adj_3D_rail))
    print("2Dc-3Ds:", get_bscan_mse(std_2D_circles, adj_3D_spheres))
    print("2Dc-2Ds:", get_bscan_mse(std_2D_circles, adj_2D_spheres))
    print()

    print("3Dc-3Dr:", get_bscan_mse(adj_3D, adj_3D_rail))
    print("3Dc-3Ds:", get_bscan_mse(adj_3D, adj_3D_spheres))
    print("3Dc-2Ds:", get_bscan_mse(adj_3D, adj_2D_spheres))
    print()
    
    print("3Dr-3Ds:", get_bscan_mse(adj_3D_rail, adj_3D_spheres))
    print("3Dr-2Ds:", get_bscan_mse(adj_3D_rail, adj_2D_spheres))
    print()

    print("3Ds-2Ds:", get_bscan_mse(adj_3D_spheres, adj_2D_spheres))
    print()

    print("====================")

    fig, axs = plt.subplots(ncols=4, nrows=2)

    titles = [["2D boxes", "2D cylinders", "3D cylinders", "3D cylinders rail"],
              ["2Dc - 2Db", "2Dc - 3Dc", "2Dc - 3Dr", "3Dc - 3Dr"]]
    data = [[std_2D_box, std_2D_circles, std_3D_cylinders, std_3D_cylinders_rail],
            [std_2D_circles - adj_box, std_2D_circles - adj_3D, std_2D_circles - adj_3D_rail, adj_3D - adj_3D_rail]]

    for i, (titles, bscans) in enumerate(zip(titles, data)):
        for j, (title, bscan) in enumerate(zip(titles, bscans)):
            plot_bscan(title, axs[i][j], bscan, dt2)

    plt.show()

if __name__ == "__main__":
    initial_comparison()