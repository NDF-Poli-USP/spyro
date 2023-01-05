#!/bin/bash
for p in {2..4}; do
	for i in {0..6}; do
		mpiexec -n 1 python demos/run_forward_acoustic_moving_mesh_marmousi_small.py $p $i
	done
done
