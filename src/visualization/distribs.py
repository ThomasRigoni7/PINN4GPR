import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
        'figure.figsize': (8, 6),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


def show_beta_distribs():
    x = np.linspace(0, 1, 100)

    beta2_2 = beta(2, 2)
    plt.plot(x, beta2_2.pdf(x))
    plt.ylabel("probability density")
    plt.xlabel("value")
    plt.show()

    beta12_25 = beta(1.2, 2.5)
    plt.plot(x, beta12_25.pdf(x))
    plt.ylabel("probability density")
    plt.xlabel("value")
    plt.show()

def show_ballast_distributions():
    """
    Shows a comparison histogram between clean and fouled ballast.
    """
    from src.dataset_creation.ballast_simulation import BallastSimulation

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
            'figure.figsize': (16, 9),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)

    clean_ballast_radii_distrib = BallastSimulation.get_clean_ballast_radii_distrib()
    fouled_ballast_radii_distrib = BallastSimulation.get_fouled_ballast_radii_distrib()

    print(clean_ballast_radii_distrib)
    print(fouled_ballast_radii_distrib)

    positions = np.flip(clean_ballast_radii_distrib[:, 1]) * 2
    width = np.flip(clean_ballast_radii_distrib[:, 0] - clean_ballast_radii_distrib[:, 1]) * 2
    height_clean = np.flip(clean_ballast_radii_distrib[:, 2])
    height_fouled = np.flip(fouled_ballast_radii_distrib[:, 2])
    print(positions)
    print(width)

    # plt.plot(x, pdf)
    plt.bar(x=positions, height=height_clean, width=width, label="clean ballast", align='edge', alpha=1)
    plt.bar(x=positions, height=height_fouled, width=width, label="fouled ballast", align='edge', alpha=0.7)
    plt.title("Ballast diameter")
    plt.xlabel("diameter (m)")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    show_ballast_distributions()
