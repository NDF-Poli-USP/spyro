#!/bin/bash
for p in {2..4}; do
	for i in {0..2}; do
		mpiexec -n 4 python demos/run_forward_acoustic_moving_mesh_marmousi_small.py $p $i
	done
done
