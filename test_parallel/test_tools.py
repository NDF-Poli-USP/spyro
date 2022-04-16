from numbers import Real
import pytest
import numpy as np
import math
import spyro

def tetrahedral_volume(p1, p2, p3, p4):
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3
    (x4, y4, z4) = p4

    A = np.array([x1, y1, z1])
    B = np.array([x2, y2, z2])
    C = np.array([x3, y3, z3])
    D = np.array([x4, y4, z4])

    volume = abs(1.0 / 6.0 * (np.dot(B - A, np.cross(C - A, D - A))))

    return volume

def triangle_area(p1, p2, p3):
    """Simple function to calculate triangle area based on its 3 vertices."""
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3

    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def test_grid_calc2d():
    grid_point_calculator_parameters = {
        ## Experiment parameters
        'source_frequency' : 5.0, # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain' :  1.429, # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'homogeneous', # Either or heterogeneous. If heterogeneous is 
        #chosen be careful to have the desired velocity model below.
        'velocity_model_file_name': 'vel_z6.25m_x12.5m_exact.segy',
        'FEM_method_to_evaluate' : 'KMV', # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension' : 2, # Domain dimension. Either 2 or 3.
        'receiver_setup' : 'near', #Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        ## Line search parameters
        'reference_degree': 4, # Degree to use in the reference case (int)
        'G_reference': 8.0, # grid point density to use in the reference case (float)
        'desired_degree': 4, # degree we are calculating G for. (int)
        'G_initial': 7.0, # Initial G for line search (float)
        'accepted_error_threshold': 0.07, 
        'g_accuracy': 1e-1
        }


    G = spyro.tools.minimum_grid_point_calculator(grid_point_calculator_parameters)
    inside = (6.9< G and G<8.0)
    print(G)
    assert inside


if __name__ == "__main__":
    test_grid_calc2d()