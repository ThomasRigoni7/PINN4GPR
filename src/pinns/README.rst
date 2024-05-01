This module contains PyTorch implementations of physics-informed neural networks with increasing levels of complexity.


The ``paper`` folder contains the implementation of some of the results shown in the paper 
`*Ground-penetrating radar wavefield simulation via physics-informed neural network solver* <https://library.seg.org/doi/10.1190/geo2022-0293.1>`_.
These include a uniform and a two-layer geometry experiments, where PINN models were mostly able to predict the wavefields accurately.

The ``rail_sample_mlp.py`` file implements an MLP-based PINN for the prediction of the full wavefield of the electric field 
in a realistic railway track sample. The models' training tends to be very unstable and often leads to predictions close to 0 everywhere.
Even when the models partially fit the data, they don't capture all of the reflections, in particular the ones with lower amplitude.

The ``time2image.py`` file implements a CNN-based PINN with the same objective. The PINN version of this architecture is 
not able to predict the expanding wave behaviour of the wavefield, but it only memorizes the observation points. This is
in contrast with what is expected from physics-informed models, so we devised a further experiment to 

The ``time2sequence.py`` file trains three different physics-informed models on a simple 1D wave propagation problem. 
These are an MLP, a CNN and an MLP with discrete output. It is shown that models with discrete output perform poorly 
on both domain extension and sparse reconstruction compared with the regular MLP PINN models. This is a possible future research
direction.

The ``models`` folder contains the model architectures, while the ``other_experiments`` folder contains further applications of the 
same and different architectures that were developed during this work.
