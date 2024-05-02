"""
This module contains a field visualization function
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def save_field_animation(field: np.ndarray, output_path: str | Path, bound_mult_factor: float = 0.2, interval: float = 400):
    """
    Shows and saves to file an animation of the provided field over time.

    Parameters
    ----------
    field : np.ndarray of shape [n_timesteps, height, width]
        ndarray containing the field values
    output_path : str | Path
        path in which to save the field animation. If None, then only shows it.
    bound_mult_factor : float, default: 0.2
        multiplication factor to apply to the min/max values for the visualization.
    interval : float, default: 400
        time interval between frames to use for the animation.
    """
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    ims = []
    for j, e_field in enumerate(field):
        im = ax.imshow(e_field, cmap='seismic',
                vmin=-np.absolute(field).max() * bound_mult_factor, vmax=np.absolute(field).max() * bound_mult_factor, animated=True)
        if j == 0:
            ax.imshow(e_field, cmap='seismic',
                vmin=-np.absolute(field).max() * bound_mult_factor, vmax=np.absolute(field).max() * bound_mult_factor, animated=True)
        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)
    plt.colorbar(im)
    if output_path is not None: 
        ani.save(output_path)
    else:
        plt.show()


if __name__ == "__main__":
    snapshots = np.load("gprmax_output_files/scan_0000/snapshots.npz")
    save_field_animation(snapshots["0001_E"], None)