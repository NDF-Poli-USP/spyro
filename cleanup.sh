#!/bin/bash
rm -f *.msh
rm -f *.vtk
rm -f *.png
rm -f *.vtu
rm -f *.pvtu
rm -f *.pvd
rm -f *.npy
rm -f *.pdf
rm -f *.dat
rm -f *.segy
rm -f *.hdf5

# Remove test outputs
rm -f results/*.vtu
rm -f results/*.pvd
rm -f results/*.pvtu
rm -f shots/*.dat

# Remove coverage files
#rm -f .coverage
#rm -f .coverage.*

# Remove generated directories
rm -rf velocity_models/test*
rm -rf results/shot*
rm -rf results/gradient
rm -rf results/*
rm -rf control_*/
rm -rf gradient*
rm -rf initial_velocity_model/
rm -rf output*
rm -rf asn*
rm -rf bsn*
rm -rf vp_end*
rm -rf test_debug*
