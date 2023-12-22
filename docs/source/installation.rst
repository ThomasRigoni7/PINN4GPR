Installation
============

Prerequisites
-------------

The project requires an installation of python 3.10 or newer. GprMax requires an installation of a C compiler like gcc. 
More info at http://docs.gprmax.com/en/latest/include_readme.html. 

Quick install
-------------

By far the easiest way to install all the required dependencies for the project is to use the provided ``install.sh`` 
bash script found in the root directory of the repository. The script requires the command ``python3`` to point at the 
desired installation of python. If not, just execute all the commands as in the script and substitute the first 
``python3`` command with the path to your desired python executable.

First, clone the repository to your machine with:

.. code-block:: bash

    git clone git@github.com:ThomasRigoni7/PINN4GPR.git

move inside the root directory:

.. code-block:: bash

    cd PINN4GPR

and execute the script:

.. code-block:: bash

    source install.sh

This will:

* Create a virtual environment inside the .venv folder in the working directory
* Clone a fork of `gprMax <https://www.gprmax.com/>`_ from https://github.com/ThomasRigoni7/PINN4GPR into ```gprmax_repo```
* Install all the required packages into the virtual environment, including building and installing 
  `gprMax <https://www.gprmax.com/>`_, then source it into the current terminal session.

At this point, if you want to run gprMax with a GPU, install the ``pycuda`` package:

.. code-block:: bash

   pip install pycuda==2023.1

For future sessions, just activate the environment with the standard

.. code-block:: bash

    source .venv/bin/activate

and deactivate with 

.. code-block:: bash

    deactivate