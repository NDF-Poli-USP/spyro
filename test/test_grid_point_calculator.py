import math
from copy import deepcopy

import pytest
from firedrake import *

#import spyro

## When testing locally:
#import sys
#sys.path.append('/home/alexandre/Development/NewEarthdrake/NewEarthdrake')
import spyro

def test_mesh_generation():
    # input parameters for test
    method = 'KMV'
    degree = 3
    G=4
    frequency = 10.
    minimum_mesh_velocity = 1.0
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity)
    
    mesh = spyro.tools.generate_mesh(model, G)
    return True

def test_wave_solver():
    pytest.skip("Model is too big for CircleCI resources. Test it locally.")
    method = 'KMV'
    degree = 3
    G=5
    frequency = 10.
    minimum_mesh_velocity = 1.0
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity)    

    spyro.tools.wave_solver(model, G)

    return True

def test_error_calc():
    pytest.skip("Model is too big for CircleCI resources. Test it locally.")
    method = 'KMV'
    degree = 2
    frequency = 10.
    minimum_mesh_velocity = 1.0
    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity)  

    G = 12
    p_exact = spyro.tools.wave_solver(model, G)
    G = 5
    p0 = spyro.tools.wave_solver(model, G)
    error = spyro.tools.error_calc(p_exact, p0, model)
    print(error)

    return True
#def test_grid_point_calculator():

    
if __name__ == "__main__":
    test_mesh_generation()
    test_wave_solver()
    test_error_calc()
