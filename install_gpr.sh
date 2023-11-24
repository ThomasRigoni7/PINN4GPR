#!/bin/bash

# clone the gprMax repository
git clone git@github.com:gprMax/gprMax.git gprmax_repo

# create and activate the virtualenv
virtualenv .gpr_venv
source .gpr_venv/bin/activate

# install the packages necessary to install gprmax
pip install -r requirements_base.txt
cd gprmax_repo

# build and install gprmax
python setup.py build
python setup.py install

# install additional packages
cd ..
pip install -r requirements.txt
