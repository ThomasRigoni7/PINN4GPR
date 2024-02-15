#!/bin/bash

# This is the script used to calculate the initial comparison B-scans, given the input files.
# The total runtime needed is around 10 hours on a system with AMD EPYC 7742 CPU and NVIDIA TITAN RTX GPU. 
# Almost all of this time is spent running the 3D models.

# create the directories in which to store the results
mkdir -p gprmax_input_files/initial_comparison/output/2D_boxes/
mkdir -p gprmax_input_files/initial_comparison/output/2D_cylinders/
mkdir -p gprmax_input_files/initial_comparison/output/3D_cylinders/
mkdir -p gprmax_input_files/initial_comparison/output/3D_cylinders_rails/
mkdir -p gprmax_input_files/initial_comparison/output/3D_spheres/
mkdir -p gprmax_input_files/initial_comparison/output/2D_spheres/

#######################
# RUN THE SIMULATIONS #
#######################

# 2D BOXES
python -m gprMax gprmax_input_files/initial_comparison/2D_boxes.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/2D_boxes/2D_boxes
mv gprmax_input_files/initial_comparison/output/2D_boxes/2D_boxes_merged.out gprmax_input_files/initial_comparison/output/2D_boxes.out

# 2D CYLINDERS
python -m gprMax gprmax_input_files/initial_comparison/2D_cylinders.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/2D_cylinders/2D_cylinders
mv gprmax_input_files/initial_comparison/output/2D_cylinders/2D_cylinders_merged.out gprmax_input_files/initial_comparison/output/2D_cylinders.out

# 3D CYLINDERS WITHOUT RAILS
python -m gprMax gprmax_input_files/initial_comparison/3D_cylinders.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/3D_cylinders/3D_cylinders
mv gprmax_input_files/initial_comparison/output/3D_cylinders/3D_cylinders_merged.out gprmax_input_files/initial_comparison/output/3D_cylinders.out

# 3D CYLINDERS WITH RAIL
python -m gprMax gprmax_input_files/initial_comparison/3D_cylinders_rails.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/3D_cylinders_rails/3D_cylinders_rails
mv gprmax_input_files/initial_comparison/output/3D_cylinders_rails/3D_cylinders_rails_merged.out gprmax_input_files/initial_comparison/output/3D_cylinders_rails.out

# 3D SPHERES
python -m gprMax gprmax_input_files/initial_comparison/3D_spheres.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/3D_spheres/3D_spheres
mv gprmax_input_files/initial_comparison/output/3D_spheres/3D_spheres_merged.out gprmax_input_files/initial_comparison/output/3D_spheres.out

# The geometry file for the 2D spheres model is already included in the repository files and obtained as in here.
# You can also manually cut the 3D spheres geometry into a 2D slice to use for the 2D spheres, 
# for this you need to first uncomment the "#geometry_objects_write ..." line in 3D_spheres.in, then run:
# python -m gprMax gprmax_input_files/initial_comparison/3D_spheres.in -gpu --geometry-only
# python gprmax_input_files/initial_comparison/slice_3D_geom.py

# 2D speres
python -m gprMax gprmax_input_files/initial_comparison/2D_spheres.in -n 55 -gpu
python -m tools.outputfiles_merge gprmax_input_files/initial_comparison/output/2D_spheres/2D_spheres
mv gprmax_input_files/initial_comparison/output/2D_spheres/2D_spheres_merged.out gprmax_input_files/initial_comparison/output/2D_spheres.out