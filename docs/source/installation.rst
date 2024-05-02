Installation
============

Prerequisites
-------------

The project requires an installation of python 3.10 or newer. GprMax requires an installation of a C compiler like gcc. 
More info at http://docs.gprmax.com/en/latest/include_readme.html. 

Quick install
-------------

By far the easiest way to install all the required dependencies for the project is to use the provided ``install.sh`` 
bash script found in the root directory of the repository. 

First, clone the repository to your machine with:

.. code-block:: bash

    git clone git@github.com:ThomasRigoni7/PINN4GPR.git

or using https with:

.. code-block:: bash

    git clone https://github.com/ThomasRigoni7/PINN4GPR.git

then move inside the root directory:

.. code-block:: bash

    cd PINN4GPR

create and activate a virtual environment with the ``venv`` packge.

.. code-block:: bash

    *path/to/python3.10* -m venv .venv
    source .venv/bin/activate

and execute the script:

.. code-block:: bash

    ./install.sh

This will:

* Clone a fork of `gprMax <https://www.gprmax.com/>`_ from https://github.com/ThomasRigoni7/PINN4GPR into ``gprmax_repo``
* Install all the required packages into the virtual environment, including building and installing 
  `gprMax <https://www.gprmax.com/>`_.

The script will ask for a confirmation to install cuda support for gprMax: 

.. code-block:: bash

    Do you wish to install cuda support for gprMax? An existing cuda installation is required.
    1) Yes
    2) No
    #?

This will install the ``pycuda`` package. This step might fail if no cuda installation is present on the system.

For future sessions, just activate the environment with the standard

.. code-block:: bash

    source .venv/bin/activate

and deactivate with 

.. code-block:: bash

    deactivate

3D ballast simuation
--------------------

The 3D ballast simulation module ``src/dataset_creation/ballast_simulation3D.py`` requires the installation of the 
``chrono`` physics engine and the ``pychrono`` python bindings, which are not included in the standard installation. 

More information on the creation of a conda environment for pychrono can be found `here <https://api.projectchrono.org/pychrono_installation.html>`_.