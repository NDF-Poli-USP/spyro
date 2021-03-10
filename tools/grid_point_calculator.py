from mpi4py import MPI
import numpy as np
import meshio
import SeismicMesh
import firedrake as fire
import time
import spyro

def minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.01):
    
    ## Chossing parameters

    if experient_type == 'homogeneous':
        minimum_mesh_velocity = 1.0

    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experient_type, receiver_type = 'near')
    
    p_exact = wave_solver(model, G =12.0)
    p_0 = wave_solver(model, G =10.0)

    error = error_calc(p_exact, p0)

    if error > TOL:
        raise ValueError('There might be something wrong with the simulation since G = 10 fails with the defined error.')

    G = searching_for_minimum(model, p_exact, model)

    return G