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


def test_mesh_generation_for_grid_calc():
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'homogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'velocity_model_file_name': 'vel_z6.25m_x12.5m_exact.segy',
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 2,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'near',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.07,
        'g_accuracy': 1e-1
    }
    Gs = [7.0, 7.1, 7.7, 8.0]
    degree_reference = grid_point_calculator_parameters['reference_degree']

    model = spyro.tools.create_model_for_grid_point_calculation(grid_point_calculator_parameters, degree_reference)
    comm = spyro.utils.mpi_init(model)
    for G in Gs:
        model["mesh"]["meshfile"] = "meshes/2Dhomogeneous"+str(G)+".msh"
        model = spyro.tools.generate_mesh(model, G, comm)


def test_input_models_receivers():
    test1 = True  # testing if 2D receivers are inside the domain on an homogeneous case
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'homogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 2,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'near',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.05,
        'g_accuracy': 1e-1
    }
    model = spyro.tools.create_model_for_grid_point_calculation(grid_point_calculator_parameters, 4)

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    p1 = (-Real_Lz, -lx)
    p2 = (-Real_Lz, Real_Lx-lx)
    p3 = (0.0, -lx)
    p4 = (0.0, Real_Lx-lx)

    areaSquare = Real_Lz*Real_Lx

    for r in model["acquisition"]["receiver_locations"]:
        area1 = triangle_area(p1, p2, r)
        area2 = triangle_area(p1, p3, r)
        area3 = triangle_area(p3, p4, r)
        area4 = triangle_area(p2, p4, r)
        test = math.isclose((area1 + area2 + area3 + area4), areaSquare, rel_tol=1e-09)
        if test is False:
            test1 = False

    test2 = True  # For 3D case
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'homogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 3,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'near',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.05,
        'g_accuracy': 1e-1
    }
    model = spyro.tools.create_model_for_grid_point_calculation(grid_point_calculator_parameters, 4)

    # FInish volume test later
    # Lz = model["mesh"]['Lz']
    # lz = model['BCs']['lz']
    # Lx = model["mesh"]['Lx']
    # lx = model['BCs']['lx']
    # Ly = model["mesh"]['Ly']
    # ly= model['BCs']['ly']

    # Real_Lz = Lz + lz
    # Real_Lx = Lx + 2*lx
    # Real_Ly = Ly + 2*ly

    # p1 = (-Real_Lz, -lx       , -ly)
    # p2 = (-Real_Lz, -lx       , Real_Ly-ly)
    # p3 = (-Real_Lz, Real_Lx-lx, -ly)
    # p4 = (-Real_Lz, Real_Lx-lx, Real_Ly-ly)
    # p5 = (0.0     , -lx       , -ly)
    # p6 = (0.0     , -lx       , Real_Ly-ly)
    # p7 = (0.0     , Real_Lx-lx, -ly)
    # p8 = (0.0     , Real_Lx-lx, Real_Ly-ly)

    # volumeSquare = Real_Lx*Real_Ly*Real_Lz

    # for r in model["acquisition"]["receiver_locations"]:
    #     volume1 = tetrahedral_volume(p1, p2, r)
    #     volume2 = tetrahedral_volume(p1, p3, r)
    #     volume3 = tetrahedral_volume(p1, p4, r)
    #     volume4 = tetrahedral_volume(p1, p5, r)
    #     volume5 = tetrahedral_volume(p1, p6, r)
    #     volume6 = tetrahedral_volume(p1, p7, r)
    #     test = math.isclose((volume1 + volume2 + volume3 + volume4), volumeSquare, rel_tol=1e-09)
    #     if test == False:
    #         test1 = False

    assert all([test1, test2])


def test_input_models_receivers_heterogeneous():
    test1 = True  # testing if 2D receivers bins are inside the domain on an heterogeneous case
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'heterogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'velocity_model_file_name': None,
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 2,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'bins',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.05,
        'g_accuracy': 1e-1
    }
    model = spyro.tools.create_model_for_grid_point_calculation(grid_point_calculator_parameters, 4)

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    p1 = (-Real_Lz, -lx)
    p2 = (-Real_Lz, Real_Lx-lx)
    p3 = (0.0, -lx)
    p4 = (0.0, Real_Lx-lx)

    areaSquare = Real_Lz*Real_Lx

    for r in model["acquisition"]["receiver_locations"]:
        area1 = triangle_area(p1, p2, r)
        area2 = triangle_area(p1, p3, r)
        area3 = triangle_area(p3, p4, r)
        area4 = triangle_area(p2, p4, r)
        test = math.isclose((area1 + area2 + area3 + area4), areaSquare, rel_tol=1e-09)
        if test is False:
            test1 = False

    test2 = True  # testing if 2D receivers line are inside the domain on an heterogeneous case
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'heterogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'velocity_model_file_name': None,
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 2,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'line',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.05,
        'g_accuracy': 1e-1
    }
    model = spyro.tools.create_model_for_grid_point_calculation(grid_point_calculator_parameters, 4)

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    p1 = (-Real_Lz, -lx)
    p2 = (-Real_Lz, Real_Lx-lx)
    p3 = (0.0, -lx)
    p4 = (0.0, Real_Lx-lx)

    areaSquare = Real_Lz*Real_Lx

    for r in model["acquisition"]["receiver_locations"]:
        area1 = triangle_area(p1, p2, r)
        area2 = triangle_area(p1, p3, r)
        area3 = triangle_area(p3, p4, r)
        area4 = triangle_area(p2, p4, r)
        test = math.isclose((area1 + area2 + area3 + area4), areaSquare, rel_tol=1e-09)
        if test is False:
            test2 = False

    assert all([test1, test2])


def test_grid_calc2d():
    grid_point_calculator_parameters = {
        # Experiment parameters
        'source_frequency': 5.0,  # Here we define the frequency of the Ricker wavelet source
        'minimum_velocity_in_the_domain': 1.429,  # The minimum velocity present in the domain.
        # if an homogeneous test case is used this velocity will be defined in the whole domain.
        'velocity_profile_type': 'homogeneous',  # Either or heterogeneous. If heterogeneous is
        # chosen be careful to have the desired velocity model below.
        'velocity_model_file_name': 'vel_z6.25m_x12.5m_exact.segy',
        'FEM_method_to_evaluate': 'KMV',  # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
        'dimension': 2,  # Domain dimension. Either 2 or 3.
        'receiver_setup': 'near',  # Either near or line. Near defines a receiver grid near to the source,
        # line defines a line of point receivers with pre-established near and far offsets.

        # Line search parameters
        'reference_degree': 4,  # Degree to use in the reference case (int)
        'G_reference': 8.0,  # grid point density to use in the reference case (float)
        'desired_degree': 4,  # degree we are calculating G for. (int)
        'G_initial': 7.0,  # Initial G for line search (float)
        'accepted_error_threshold': 0.07,
        'g_accuracy': 1e-1
    }

    G = spyro.tools.minimum_grid_point_calculator(grid_point_calculator_parameters)
    inside = (6.9 < G and G < 8.0)
    print(G)
    assert inside


if __name__ == "__main__":
    test_mesh_generation_for_grid_calc()
