"""
This module contains various miscellaneous visualization functions
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.dataset_creation.ballast_simulation import BallastSimulation

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
    if output_path is not None: 
        ani.save(output_path)
    plt.show()

def show_ballast_distributions():
    """
    Shows a comparison histogram between clean and fouled ballast.
    """
    clean_ballast_radii_distrib = BallastSimulation.get_clean_ballast_radii_distrib()
    fouled_ballast_radii_distrib = BallastSimulation.get_fouled_ballast_radii_distrib()

    print(clean_ballast_radii_distrib)
    positions = np.hstack([clean_ballast_radii_distrib[-1, 1], clean_ballast_radii_distrib[-1, 1], np.flip(clean_ballast_radii_distrib[:, 0]), clean_ballast_radii_distrib[0, 0],])
    print(positions)
    densities = []
    for sieve in clean_ballast_radii_distrib:
        density = sieve[2] / (sieve[0]-sieve[1])
        densities.append(density)
    height_clean = np.hstack([np.flip(np.array(densities))])
    height_clean = np.hstack([0, height_clean, height_clean[-1], 0])
    densities = []
    for sieve in fouled_ballast_radii_distrib:
        density = sieve[2] / (sieve[0]-sieve[1])
        densities.append(density)
    height_fouled = np.hstack([np.flip(np.array(densities))])
    height_fouled = np.hstack([0, height_fouled, height_fouled[-1], 0])

    # plt.plot(x, pdf)
    plt.step(positions, height_clean, where="post", label="clean ballast")
    plt.step(positions, height_fouled, where="post", label="fouled ballast")
    plt.title("Ballast distribution comparison")
    plt.xlabel("radius")
    plt.ylabel("pdf")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # snapshots = np.load("gprmax_output_files/scan_0000/snapshots.npz")
    # save_field_animation(snapshots["0001_E"], None)
    show_ballast_distributions()