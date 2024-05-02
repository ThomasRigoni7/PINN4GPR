#!/bin/bash

# clone the gprMax repository
git clone https://github.com/ThomasRigoni7/gprMax.git gprmax_repo

# install the packages necessary to install gprmax
pip install -r requirements_base.txt
cd gprmax_repo

# build and install gprmax
python setup.py build
python setup.py install

# install additional packages
cd ..
pip install -r requirements.txt

echo ""
echo "Do you wish to install cuda support for gprMax? An existing cuda installation is required."
select yn in "Yes" "No"; do
    case $yn in
        Yes ) pip install pycuda==2024.1; break;;
        No ) exit;;
    esac
done
