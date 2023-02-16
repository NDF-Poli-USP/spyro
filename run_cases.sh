#!/bin/bash
for quad in {0..1}; do
	for dg in {1..1}; do # FIXME 0..1 
		for amr in {1..1}; do # FIXME 0..1
			for p in {4..4}; do
				for i in 0 1 3; do 
				#for i in {0..4}; do
					#echo $i
					#mpiexec -n 4 python demos/run_forward_acoustic_moving_mesh_camembert.py $p $i $amr $dg $quad
					#mpiexec -n 4 python demos/run_forward_acoustic_moving_mesh_generic_polygon_with_layers.py $p $i $amr $dg $quad
					#mpiexec -n 1 python demos/run_forward_elastic_moving_mesh_camembert.py $p $i $amr $dg $quad
					mpiexec -n 1 python demos/run_forward_elastic_moving_mesh_generic_polygon_with_layers.py $p $i $amr $dg $quad
				done
			done
		done
	done
done
