from mpi4py import MPI
import numpy as np
from scipy import interpolate
import meshio
import SeismicMesh
import firedrake as fire
import time
import copy
import spyro

def minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.2, G_init = 12):
    ## Chossing parameters

    if experient_type == 'homogeneous':
        minimum_mesh_velocity = 1.0

    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experient_type, receiver_type = 'near')
    comm = spyro.utils.mpi_init(model)
    
    print("Initial method check", flush = True)
    p_exact = wave_solver(model, G =G_init, comm = comm)
    comm.comm.barrier()
    
    print('Before p0 calc', flush = True)
    p_0 = wave_solver(model, G =G_init - 0.2*G_init, comm = comm)
    quit()

    print('Before error calc', flush = True)
    error = error_calc(p_exact, p_0, model, comm = comm)
    print('After error calc')

    if error > TOL:
        print(error)
        raise ValueError('There might be something wrong with the simulation since G = 10 fails with the defined error.')

    print("Searching for minimum")
    G = searching_for_minimum(model, p_exact, TOL, starting_G=G_init - 0.2*G_init, comm = comm)

    return G

def wave_solver(model, G, comm = False):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']

    mesh = generate_mesh(model, G, comm)
    
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    vp_exact = fire.Constant(minimum_mesh_velocity)

    new_dt = 0.2*spyro.estimate_timestep(mesh, V, vp_exact)

    model['timeaxis']['dt'] = comm.comm.allreduce(new_dt, op=MPI.MIN)
    if comm.comm.rank == 0:
        print(
            f"Maximum stable timestep used is: {model['timeaxis']['dt']} seconds",
            flush=True,
        )

    sources = spyro.Sources(model, mesh, V, comm).create()
    receivers = spyro.Receivers(model, mesh, V, comm).create()

    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            t1 = time.time()
            p_field, p_recv = spyro.solvers.Leapfrog(
                model, mesh, comm, vp_exact, sources, receivers, source_num=sn, output= True
            )
            print(time.time() - t1)

    return p_recv

def generate_mesh(model,G, comm):
    print('Entering mesh generation', flush = True)
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

    if comm.comm.rank == 0:
        points, cells = SeismicMesh.generate_mesh(
        domain=rec, 
        edge_length=edge_length, 
        mesh_improvement = False,
        comm = comm.ensemble_comm,
        verbose = 0
        )
        print('entering spatial rank 0 after mesh generation')
        points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
        a=np.amin(SeismicMesh.geometry.simp_qual(points, cells))
        if model['testing_parameters']['experiment_type'] == 'heterogenous':
            points, cells = SeismicMesh.geometry.laplacian2(points, cells)
        meshio.write_points_cells("homogeneous"+str(G)+".msh",
            points,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("homogeneous"+str(G)+".vtk",
            points,[("triangle", cells)],
            file_format="vtk"
            )

    comm.comm.barrier()
    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "homogeneous"+str(G)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

    print('Finishing mesh generation', flush = True)
    return mesh

def searching_for_minimum(model, p_exact, TOL, accuracy = 0.1, starting_G = 10.0, comm=False):
    error = 0.0
    G = starting_G

    # fast loop
    while error < TOL:
        dif = max(G*0.2, accuracy)
        G = G - dif
        print('With G equal to '+str(G) )
        p0 = wave_solver(model,G, comm)
        error = error_calc(p_exact, p0, model, comm = comm)
        print('Error of '+str(error))

    G += dif
    # slow loop
    if dif > accuracy :
        error = 0.0
        while error < TOL:
            dif = accuracy
            G = G - dif
            print('With G equal to '+str(G) )
            p0 = wave_solver(model,G, comm )
            error = error_calc(p_exact, p0, model, comm = comm)
            print('Error of '+str(error))

        G+= dif

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

def error_calc(p_exact, p, model, comm = False):

    #comm = spyro.utils.mpi_init(model)
    times, receivers = p.shape
    dt = model["timeaxis"]['tf']/times

    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    p_exact = time_interpolation(p_exact, p, model)
    p_diff = p_exact-p
    max_diff = 0.0

    if comm.ensemble_comm.rank ==0:
        numerator = 0.0
        denominator = 0.0
        for receiver in range(receivers):
            numerator_time_int = 0.0
            denominator_time_int = 0.0
            for t in range(times-1):
                numerator_time_int   += (p_exact[t,receiver]-p[t,receiver])**2*dt
                denominator_time_int += (p_exact[t,receiver])**2*dt

                # if p_exact[t,receiver] > 1e-10:
                #     diff = abs(p_exact[t,receiver]-p[t,receiver])/p_exact[t,receiver]
                #     if diff > 1e-9 and diff > max_diff:
                #         max_diff = copy.deepcopy(diff)

            numerator   += numerator_time_int
            denominator += denominator_time_int

        error = np.sqrt(numerator/denominator)

    if numerator < 1e-15:
        print('Warning: error too small to measure correctly.')
        error = 0.0
    if denominator < 1e-15:
        print("Warning: receivers don't appear to register a shot.")
        error = 0.0

    print("ERROR IS ")
    print(error)
    print("Maximum percentage difference ")
    print(max_diff)
    return error

def time_interpolation(p_old, p_exact, model):
    times, receivers = p_exact.shape
    dt = model["timeaxis"]['tf']/times

    times_old, rec = p_old.shape
    dt_old = model["timeaxis"]['tf']/times_old
    time_vector_old = np.zeros((1,times_old))
    for ite in range(times_old):
        time_vector_old[0,ite] = dt_old*ite

    time_vector_new = np.zeros((1,times))
    for ite in range(times):
        time_vector_new[0,ite] = dt*ite

    p = np.zeros((times, receivers))
    for receiver in range(receivers):
        f = interpolate.interp1d(time_vector_old[0,:], p_old[:,receiver] )
        p[:,receiver] = f(time_vector_new[0,:])

    return p