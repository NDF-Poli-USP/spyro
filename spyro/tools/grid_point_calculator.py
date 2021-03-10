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

def wave_solver(model, G):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    comm = spyro.utils.mpi_init(model)

    mesh = generate_mesh(model, G)
    
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    vp_exact = fire.Constant(minimum_mesh_velocity)

    sources = spyro.Sources(model, mesh, V, comm).create()
    receivers = spyro.Receivers(model, mesh, V, comm).create()

    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            t1 = time.time()
            p_field, p_recv = spyro.solvers.Leapfrog(
                model, mesh, comm, vp_exact, sources, receivers, source_num=sn
            )
            print(time.time() - t1)

    return p_recv

def generate_mesh(model,G):
    M = grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency
    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['PML']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['PML']['lx']
    pml_fraction = lx/Lx

    Real_Lz = Lz + 2*lz
    Real_Lx = Lx + lx
    edge_length = lbda/M

    bbox = (0.0, Real_Lz, 0.0, Real_Lx)
    rec = SeismicMesh.Rectangle(bbox)
    
    points, cells = SeismicMesh.generate_mesh(
        domain=rec, 
        edge_length=edge_length, 
        mesh_improvement = False
        )
    meshio.write_points_cells("homogeneous"+str(G)+".msh",
        points,[("triangle", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("homogeneous"+str(G)+".vtk",
        points,[("triangle", cells)],
        file_format="vtk", 
        binary = False
        )

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "homogeneous"+str(G)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

    return mesh

def searching_for_minimum(model, p_exact, TOL, accuracy = 0.1, starting_G = 5.0):
    error = 0.0
    G = starting_G

    # fast loop
    while error < TOL:
        dif = max(G*0.2, accuracy)
        G = G - dif
        print(G)
        wave_solver(model,G)
        error = error_calc(p_exact, p0, model)
        print(error)

    G += diff
    # slow loop
    if diff > accuracy :
        error = 0.0
        while error < TOL:
            dif = accuracy
            G = G - dif
            print(G)
            wave_solver(model,G)
            error = error_calc(p_exact, p0, model)
            print(error)

        G+= diff

    return G

def grid_point_to_mesh_point_converter_for_seismicmesh(model, G):
    degree = model["opts"]['degree']
    if model["opts"]["method"] == 'KMV':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.2934695559090401*G
        if degree == 4:
            M = 0.21132486540518713*G
        if degree == 5:
            M = 0.20231237605867816*G

    if model["opts"]["method"] == 'CG':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.333333333333333*G
        if degree == 4:
            M = 0.25*G
        if degree == 5:
            M = 0.2*G

    if model["opts"]["method"] == 'spectral':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.27639320225002106*G
        if degree == 4:
            M = 0.32732683535398854*G
        if degree == 5:
            M = 0.23991190372440996*G

    return M

def error_calc(p_exact, p0, model):

    times, receivers = p_exact.shape
    dt = model["timeaxis"]['tf']/times

    numerator = 0.0
    denominator = 0.0
    for receiver in range(receivers):
        numerator_time_int = 0.0
        denominator_time_int = 0.0
        for time in range(times):
            numerator_time_int   += (p_exact[time,receiver]-p0[time,receiver])**2*dt
            denominator_time_int += (p_exact[time,receiver])**2*dt

        numerator   += numerator_time_int
        denominator += denominator_time_int

    error = np.sqrt(numerator/denominator)

    if numerator < 1e-15:
        print('Warning: error too small to measure correctly.')
        error = 0.0

    return error