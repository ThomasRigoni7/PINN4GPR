#!/bin/bash

# install the forked version of the gprMax repository
git clone git@github.com:ThomasRigoni7/gprMax.git gprmax_repo
virtualenv .gpr_venv
source .gpr_venv/bin/activate
pip install -r requirements_base.txt
cd gprmax_repo
python setup.py build
python setup.py install
cd ..
pip install -r requirements.txt
