from mpi4py import MPI
import numpy as np
from scipy import interpolate
import meshio
import SeismicMesh
import firedrake as fire
import time
import copy
import spyro

def minimum_grid_point_calculator(frequency, method, degree, experiment_type = 'homogeneous', TOL = 0.2, G_init = 12, receiver_type = 'near'):
    """ Function to calculate necessary grid point density.

    Parameters
    ----------
    frequency: `float`
        Source frequency to use in calculation
    method: `string`
        The finite element method choosen
    degree: `int`
        Polynomial degree of finite element space
    experiment_type: `string`
        Only options are `homogenous` or `heterogenous`
    TOL: `float`
        Error threshold permited on minimum grid density
    G_init: `float`
        Initial grid density value to begin search

    Returns
    -------
    G: `float`
        Minimum grid point density necessary for a `experiment_type` mesh with a FEM whith 
        the degree and method specified within the specified error tolerance
        
    """
    
    ## Chossing parameters

    start_time= time.time()
    print("Starting initial method check", flush = True)

    if experiment_type == 'homogeneous':
        minimum_mesh_velocity = 1.429
    elif experiment_type == 'heterogenous':
        minimum_mesh_velocity = False # This variable isnt needed in heterogenous models because of seismicmesh

    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = 'near')
    #print("Model built at time "+str(time.time()-start_time), flush = True)
    comm = spyro.utils.mpi_init(model)
    #print("Comm built at time "+str(time.time()-start_time), flush = True)
    
    p_exact = wave_solver(model, G =G_init, comm = comm)
    print("p_exact finished at time "+str(time.time()-start_time), flush = True)
    p_0 = wave_solver(model, G =G_init - 0.2*G_init, comm = comm)
    print("p_0 finished at time "+str(time.time()-start_time), flush = True)
    #quit()

    comm.comm.barrier()
    error = error_calc(p_exact, p_0, model, comm = comm)
    print("error calc at time "+str(time.time()-start_time), flush = True)

    if error > TOL:
        print(error)
        raise ValueError('There might be something wrong with the simulation since G = 10 fails with the defined error.')

    print("Entering search at time "+str(time.time()-start_time), flush = True)
    #print("Searching for minimum")
    G = searching_for_minimum(model, p_exact, TOL, starting_G=G_init - 0.2*G_init, comm = comm)

    return G

def wave_solver(model, G, comm = False, mesh_generation = True):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']

    mesh = generate_mesh(model, G, comm)

    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    if model['testing_parameters']['experiment_type'] == 'homogeneous':
        vp_exact = fire.Constant(minimum_mesh_velocity)
    elif model['testing_parameters']['experiment_type'] == 'heterogeneous':
        vp_exact = spyro.io.interpolate(model, mesh, V, guess=False)

    if model["opts"]["method"] == 'KMV':
        estimate_max_eigenvalue=True
    else:
        estimate_max_eigenvalue=False
    new_dt = 0.2*spyro.estimate_timestep(mesh, V, vp_exact,estimate_max_eigenvalue=estimate_max_eigenvalue)

    model['timeaxis']['dt'] = comm.comm.allreduce(new_dt, op=MPI.MIN)
    if model['timeaxis']['dt'] > 0.001:
        model['timeaxis']['dt'] = 0.001
    if comm.comm.rank == 0:
        print(
            f"Maximum stable timestep is: {comm.comm.allreduce(new_dt, op=MPI.MIN)} seconds",
            flush=True,
        )
        print(
            f"Maximum stable timestep used is: {model['timeaxis']['dt']} seconds",
            flush=True,
        )

    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
                                        dt=model["timeaxis"]["dt"],
                                        tf=model["timeaxis"]["tf"],
                                        freq=model["acquisition"]["frequency"],
                                    )

    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            t1 = time.time()
            p_field, p_recv = spyro.solvers.forward(
                model, mesh, comm, vp_exact, sources, wavelet, receivers, source_num=sn, output= False)
            print(time.time() - t1)

    return p_recv

def generate_mesh(model,G, comm):
    if model["opts"]["dimension"] == 2:
        mesh = generate_mesh2D(model,G, comm)
    elif model["opts"]["dimension"] == 3:
        mesh = generate_mesh3D(model,G, comm)
    else:
        raise ValueError("Wrong dimension in input model.")
    return mesh

def searching_for_minimum(model, p_exact, TOL, accuracy = 0.1, starting_G = 10.0, comm=False):
    print("Search began, time reset", flush = True)
    start_time = time.time()
    error = 0.0
    G = starting_G

    # fast loop
    print("Entering fast loop at time "+str(time.time()-start_time), flush = True)
    while error < TOL:
        dif = max(G*0.1, accuracy)
        G = G - dif
        print('With G equal to '+str(G) )
        print("Entering wave solver at time "+str(time.time()-start_time), flush = True)
        p0 = wave_solver(model,G, comm)
        print("Entering error calc at time "+str(time.time()-start_time), flush = True)
        error = error_calc(p_exact, p0, model, comm = comm)
        print('Error of '+str(error))

    G += dif
    # slow loop
    print("Entering slow loop at time "+str(time.time()-start_time), flush = True)
    if dif > accuracy :
        error = 0.0
        while error < TOL:
            dif = accuracy
            G = G - dif
            print('With G equal to '+str(G) )
            print("Entering wave solver at time "+str(time.time()-start_time), flush = True)
            p0 = wave_solver(model,G, comm )
            print("Entering error calc at time "+str(time.time()-start_time), flush = True)
            error = error_calc(p_exact, p0, model, comm = comm)
            print('Error of '+str(error))

        G+= dif

    return G

#UPDATE THIS
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
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    times_p_exact, r_p_exact = p_exact.shape
    times_p, r_p = p.shape
    if times_p_exact > times_p: #then we interpolate p_exact
        times, receivers = p.shape
        dt = model["timeaxis"]['tf']/times
        p_exact = time_interpolation(p_exact, p, model)
    elif times_p_exact < times_p: #then we interpolate p
        times, receivers = p_exact.shape
        dt = model["timeaxis"]['tf']/times
        p = time_interpolation(p, p_exact, model)
    else: #then we dont need to interpolate
        times, receivers = p.shape
        dt = model["timeaxis"]['tf']/times
    #p = time_interpolation(p, p_exact, model)

    p_diff = p_exact-p
    max_absolute_diff = 0.0
    max_percentage_diff = 0.0

    if comm.ensemble_comm.rank ==0:
        numerator = 0.0
        denominator = 0.0
        for receiver in range(receivers):
            numerator_time_int = 0.0
            denominator_time_int = 0.0
            for t in range(times-1):
                top_integration = (p_exact[t,receiver]-p[t,receiver])**2*dt
                bot_integration = (p_exact[t,receiver])**2*dt

                # Adding 1e-25 filter to receivers to eliminate noise
                numerator_time_int   += top_integration

                denominator_time_int += bot_integration


                diff = p_exact[t,receiver]-p[t,receiver]
                if abs(diff) > 1e-15 and abs(diff) > max_absolute_diff:
                    max_absolute_diff = copy.deepcopy(diff)
                
                if abs(diff) > 1e-15 and abs(p_exact[t,receiver]) > 1e-15:
                    percentage_diff = abs( diff/p_exact[t,receiver]  )*100
                    if percentage_diff > max_percentage_diff:
                        max_percentage_diff = copy.deepcopy(percentage_diff)

            numerator   += numerator_time_int
            denominator += denominator_time_int
	
    if denominator > 1e-15:
        error = np.sqrt(numerator/denominator)

    # if numerator < 1e-15:
    #     print('Warning: error too small to measure correctly.', flush = True)
    #     error = 0.0
    if denominator < 1e-15:
        print("Warning: receivers don't appear to register a shot.", flush = True)
        error = 0.0

    # print("ERROR IS ", flush = True)
    # print(error, flush = True)
    # print("Maximum absolute error ", flush = True)
    # print(max_absolute_diff, flush = True)
    # print("Maximum percentage error ", flush = True)
    # print(max_percentage_diff, flush = True)
    return error

def error_calc_line(p_exact, p, model, comm = False):
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    times_p_exact, = p_exact.shape
    times_p, = p.shape
    if times_p_exact > times_p: #then we interpolate p_exact
        times,= p.shape
        dt = model["timeaxis"]['tf']/times
        p_exact = time_interpolation_line(p_exact, p, model)
    elif times_p_exact < times_p: #then we interpolate p
        times,= p_exact.shape
        dt = model["timeaxis"]['tf']/times
        p = time_interpolation_line(p, p_exact, model)
    else: #then we dont need to interpolate
        times, = p.shape
        dt = model["timeaxis"]['tf']/times


    if comm.ensemble_comm.rank ==0:
        numerator_time_int = 0.0
        denominator_time_int = 0.0
        # Integrating with trapezoidal rule
        for t in range(times-1):
            numerator_time_int   += (p_exact[t]-p[t])**2
            denominator_time_int += (p_exact[t])**2
        numerator_time_int -= ((p_exact[0]-p[0])**2 + (p_exact[times-1]-p[times-1])**2)/2
        numerator_time_int *= dt
        denominator_time_int -= (p_exact[0]**2+p_exact[times-1]**2)/2
        denominator_time_int *= dt
	
        #if denominator_time_int > 1e-15:
        error = np.sqrt(numerator_time_int/denominator_time_int)

        #if numerator_time_int < 1e-15:
         #   print('Warning: error too small to measure correctly.', flush = True)
            #error = 0.0
        if denominator_time_int < 1e-15:
            print("Warning: receivers don't appear to register a shot.", flush = True)
            error = 0.0

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

def time_interpolation_line(p_old, p_exact, model):
    times, = p_exact.shape
    dt = model["timeaxis"]['tf']/times

    times_old, = p_old.shape
    dt_old = model["timeaxis"]['tf']/times_old
    time_vector_old = np.zeros((1,times_old))
    for ite in range(times_old):
        time_vector_old[0,ite] = dt_old*ite

    time_vector_new = np.zeros((1,times))
    for ite in range(times):
        time_vector_new[0,ite] = dt*ite

    p = np.zeros((times,))
    f = interpolate.interp1d(time_vector_old[0,:], p_old[:] )
    p[:] = f(time_vector_new[0,:])

    return p

def generate_mesh2D(model,G, comm):

    print('Entering mesh generation', flush = True)
    M = grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    disk_M = grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    if model['testing_parameters']['experiment_type']== 'homogeneous':
        minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
        frequency = model["acquisition"]['frequency']
        lbda = minimum_mesh_velocity/frequency

        Real_Lz = Lz + lz
        Real_Lx = Lx + 2*lx
        edge_length = lbda/M

        bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
        rec = SeismicMesh.Rectangle(bbox)

        if comm.comm.rank == 0:
            # Creating rectangular mesh
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

            meshio.write_points_cells("meshes/2Dhomogeneous"+str(G)+".msh",
                points,[("triangle", cells)],
                file_format="gmsh22", 
                binary = False
                )
            meshio.write_points_cells("meshes/2Dhomogeneous"+str(G)+".vtk",
                points,[("triangle", cells)],
                file_format="vtk"
                )

        comm.comm.barrier()
        if method == "CG" or method == "KMV":
            mesh = fire.Mesh(
                "meshes/2Dhomogeneous"+str(G)+".msh",
                distribution_parameters={
                    "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
                },
            )

    elif model['testing_parameters']['experiment_type']== 'heterogeneous':
        # Name of SEG-Y file containg velocity model.
        fname = "vel_z6.25m_x12.5m_exact.segy"

        # Bounding box describing domain extents (corner coordinates)
        bbox = (-12000.0, 0.0, 0.0, 67000.0)

        rectangle =SeismicMesh.Rectangle(bbox)

        # Desired minimum mesh size in domain
        frequency = model["acquisition"]['frequency']
        hmin = 1429.0/(M*frequency)

        # Construct mesh sizing object from velocity model
        ef = SeismicMesh.get_sizing_function_from_segy(
            fname,
            bbox,
            hmin=hmin,
            wl=M,
            freq=5.0,
            grade=0.15,
            domain_pad=model["BCs"]["lz"],
            pad_style="edge",
        )

        points, cells = SeismicMesh.generate_mesh(domain=rectangle, edge_length=ef, verbose = 0, mesh_improvement=False )

        meshio.write_points_cells("meshes/2Dheterogeneous"+str(G)+".msh",
            points/1000,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/2Dheterogeneous"+str(G)+".vtk",
                points/1000,[("triangle", cells)],
                file_format="vtk"
                )

        comm.comm.barrier()
        if method == "CG" or method == "KMV":
            mesh = fire.Mesh(
                "meshes/2Dheterogeneous"+str(G)+".msh",
                distribution_parameters={
                    "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
                },
            )
    print('Finishing mesh generation', flush = True)
    return mesh

def generate_mesh3D(model, G, comm):

    print('Entering mesh generation', flush = True)
    M = grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']
    Ly = model["mesh"]['Ly']
    ly= model['BCs']['ly']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    Real_Ly = Ly + 2*ly

    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    edge_length = lbda/M
    #print(Real_Lz)

    bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx, -ly, Real_Ly-ly)
    cube = SeismicMesh.Cube(bbox)

    if comm.comm.rank == 0:
        # Creating rectangular mesh
        points, cells = SeismicMesh.generate_mesh(
        domain=cube, 
        edge_length=edge_length, 
        mesh_improvement = False,
        max_iter = 75,
        comm = comm.ensemble_comm,
        verbose = 0
        )

        points, cells = SeismicMesh.sliver_removal(points=points, bbox=bbox, domain=cube, edge_length=edge_length, preserve=True, max_iter=200)
    
        print('entering spatial rank 0 after mesh generation')
        
        meshio.write_points_cells("meshes/3Dhomogeneous"+str(G)+".msh",
            points,[("tetra", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/3Dhomogeneous"+str(G)+".vtk",
            points,[("tetra", cells)],
            file_format="vtk"
            )

    comm.comm.barrier()
    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/3Dhomogeneous"+str(G)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

    
    print('Finishing mesh generation', flush = True)
    return mesh

def mesh_generation(model, Gs, comm):
    for G in Gs:
        mesh = generate_mesh(model, G, comm)
    
    return True