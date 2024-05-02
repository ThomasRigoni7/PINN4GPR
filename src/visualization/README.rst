This folder contains various scripts used for visualization purposes.

The ``dataset_plots.py`` file creates plots for samples with specific features in the dataset.

The ``distribs.py`` file shows plots for the beta distributions used in this work, together with clean and fouled ballast distributions.

The ``field.py`` module contains a field visualization function.

The ``initial_comparison.py`` file creates all the visualizations related to the initial simulated model comparison
performed as an initial step for dataset generation. Both 2D and 3D geometries are considered and the difference between
subsequent models is evaluated.

The ``model_predictions.py`` file contains the code used to generate all the figures related to the NN models.

The ``show_field_amplitudes.py`` file creates a visualization of the mean maximum amplitude 
for the samples in the dataset with respect to time and fits curves to it. This was used 
in the implementation of a time-dependant weight for the networks, which did not significantly improve the results.