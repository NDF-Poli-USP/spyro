#!/bin/bash
for p in {2..3}; do
	for i in {1..3}; do
		mpiexec -n 2 python demos/run_forward_acoustic_moving_mesh_marmousi_small.py $p $i
	done
done
