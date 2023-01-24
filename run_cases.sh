#!/bin/bash
for quad in {0..1}; do
	for dg in {0..1}; do 
		for amr in {0..1}; do
			for p in {4..4}; do
				for i in {0..4}; do
					#echo $i
					#mpiexec -n 4 python demos/run_forward_acoustic_moving_mesh_camembert.py $p $i $amr $dg $quad
					mpiexec -n 4 python demos/run_forward_acoustic_moving_mesh_generic_polygon_with_layers.py $p $i $amr $dg $quad
				done
			done
		done
	done
done
