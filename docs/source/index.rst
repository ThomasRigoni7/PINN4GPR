.. PINN4GPR documentation master file, created by
   sphinx-quickstart on Sun Nov 19 23:28:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PINN4GPR's documentation!
====================================

This project enables you to:

#. Generate Ground Penetrating Radar (GPR) datasets of realistic railway track configurations with `gprMax <https://www.gprmax.com/>`_. 
#. Train a CNN-based surrogate model for gprMax on the generated data, which at inference time is two orders of magnitude faster than FDTD simulations.
#. Use the surrogate model for faster large-scale dataset generation.
#. Explore the use of physics-informed neural networks (PINNs) for the approximation of GPR wavefield data in complex railway track geometries.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   tutorial

.. toctree:: 
   :maxdepth: 2
   :caption: API reference

   dataset_creation
   pinns
   tests
   visualization


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
