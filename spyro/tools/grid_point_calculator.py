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

    if experient_type == 'homogeneous':
        minimum_mesh_velocity = 1.429
    elif experiment_type == 'heterogenous':
        minimum_mesh_velocity = False # This variable isnt needed in heterogenous models because of seismicmesh

    model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experient_type, receiver_type = 'near')
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

def wave_solver(model, G, comm = False):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']

    mesh = generate_mesh(model, G, comm)
    
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    spyro.sources.source_dof_finder(V, model)

    if model['testing_parameters']['experiment_type'] == 'homogeneous':
        vp_exact = fire.Constant(minimum_mesh_velocity)
    elif model['testing_parameters']['experiment_type'] == 'heterogenous':
        vp_exact = spyro.io.interpolate(model, mesh, V, guess=False)


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
                model, mesh, comm, vp_exact, sources, receivers, source_num=sn, output= True, G = G
            )
            print(time.time() - t1)

    return p_recv

def generate_mesh(model,G, comm):
    print('Entering mesh generation', flush = True)
    M = grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    disk_M = grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['PML']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['PML']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    if model['testing_parameters']['experiment_type']== 'homogeneous':
        minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
        frequency = model["acquisition"]['frequency']
        lbda = minimum_mesh_velocity/frequency

        Lz = model["mesh"]['Lz']
        lz = model['PML']['lz']
        Lx = model["mesh"]['Lx']
        lx = model['PML']['lx']
        pml_fraction = lx/Lx

        Real_Lz = Lz + lz
        Real_Lx = Lx + 2*lx
        edge_length = lbda/M

        bbox = (-Real_Lz, 0.0, 0.0, Real_Lx)
        rec = SeismicMesh.Rectangle(bbox)

        if comm.comm.rank == 0:
            #creating disk around source
            if model['acquisition']['source_mesh_point']:
                source_position = model['acquisition']['source_pos']
                fixed_points = source_position
            elif model['acquisition']['source_mesh_point'] == False:
                disk_M = grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
                disk = SeismicMesh.Disk([Real_Lz/2, Real_Lx/2], lbda)
                fixed_points, disk_cells = SeismicMesh.generate_mesh(
                    domain=disk,
                    edge_length=lbda/disk_M,
                    mesh_improvement = False,
                    comm = comm.ensemble_comm,
                    verbose = 0
                )

            # Creating rectangular mesh
            points, cells = SeismicMesh.generate_mesh(
            domain=rec, 
            edge_length=edge_length, 
            mesh_improvement = False,
            comm = comm.ensemble_comm,
            pfix = fixed_points,
            verbose = 0
            )
            print('entering spatial rank 0 after mesh generation')
            
            points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
            a=np.amin(SeismicMesh.geometry.simp_qual(points, cells))
            if model['testing_parameters']['experiment_type'] == 'heterogenous':
                points, cells = SeismicMesh.geometry.laplacian2(points, cells)
            meshio.write_points_cells("meshes/homogeneous"+str(G)+".msh",
                points,[("triangle", cells)],
                file_format="gmsh22", 
                binary = False
                )
            meshio.write_points_cells("meshes/homogeneous"+str(G)+".vtk",
                points,[("triangle", cells)],
                file_format="vtk"
                )

        comm.comm.barrier()
        if method == "CG" or method == "KMV":
            mesh = fire.Mesh(
                "meshes/homogeneous"+str(G)+".msh",
                distribution_parameters={
                    "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
                },
            )

    elif model['testing_parameters']['experiment_type']== 'heterogenous':
        # Name of SEG-Y file containg velocity model.
        fname = "vel_z6.25m_x12.5m_exact.segy"

        # Bounding box describing domain extents (corner coordinates)
        bbox = (-12000.0, 0.0, 0.0, 67000.0)

        rectangle =SeismicMesh.Rectangle(bbox)

        # Desired minimum mesh size in domain
        frequency = model["acquisition"]['frequency']
        hmin = 1429.0/(M*frequency)
        edge_length_disk = 1429.0/(disk_M*frequency)

        if model['acquisition']['source_mesh_point']:
            source_position = model['acquisition']['source_pos']
            z_source, x_source = source_position[0]
            disk = SeismicMesh.Disk([z_source*1000, x_source*1000], 1429.0/frequency)
            disk_points, cells = SeismicMesh.generate_mesh(domain=disk, edge_length=edge_length_disk, verbose = 0, mesh_improvement=False )
            meshio.write_points_cells("meshes/disk"+str(G)+".vtk",
                    disk_points/1000,[("triangle", cells)],
                    file_format="vtk"
                    )
            source_z , source_x = source_position[0]
            source_points = [(source_z*1000.0,source_x*1000.0)]
            fixed_points = np.append(disk_points,source_points, axis = 0)

        elif model['acquisition']['source_mesh_point'] == False:
            # Mesh sizing for disk
            source_pos = model['acquisition']['source_pos']
            z_source, x_source = source_pos[0]
            disk = SeismicMesh.Disk([z_source*1000, x_source*1000], 1429.0/frequency)
            fixed_points, cells = SeismicMesh.generate_mesh(domain=disk, edge_length=edge_length_disk, verbose = 0, mesh_improvement=False )
            meshio.write_points_cells("meshes/disk"+str(G)+".vtk",
                    disk_points/1000,[("triangle", cells)],
                    file_format="vtk"
                    )

        # Construct mesh sizing object from velocity model
        ef = SeismicMesh.get_sizing_function_from_segy(
            fname,
            bbox,
            hmin=hmin,
            wl=M,
            freq=5.0,
            grade=0.15,
            domain_pad=model["PML"]["lz"],
            pad_style="edge",
        )

        points, cells = SeismicMesh.generate_mesh(domain=rectangle, edge_length=ef, pfix =fixed_points, verbose = 0, mesh_improvement=False )

        #points, cells = SeismicMesh.geometry.laplacian2(points, cells)
        meshio.write_points_cells("meshes/heterogenous"+str(G)+".msh",
            points/1000,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/heterogenous"+str(G)+".vtk",
                points/1000,[("triangle", cells)],
                file_format="vtk"
                )

        comm.comm.barrier()
        if method == "CG" or method == "KMV":
            mesh = fire.Mesh(
                "meshes/heterogenous"+str(G)+".msh",
                distribution_parameters={
                    "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
                },
            )
    print('Finishing mesh generation', flush = True)
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
                        if max_percentage_diff > 10.:
                            print("Weird error "+ str(max_percentage_diff) +" on time "+str(dt*t)+" and receiver "+str(receiver), flush = True)
                            print(p_exact[t,receiver], flush = True)
                            print(p[t,receiver], flush = True)


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

    print("ERROR IS ", flush = True)
    print(error, flush = True)
    print("Maximum absolute error ", flush = True)
    print(max_absolute_diff, flush = True)
    print("Maximum percentage error ", flush = True)
    print(max_percentage_diff, flush = True)
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

def p_filter(p, tol=1e-20):
    times, receivers = p.shape
    for ti in range(times):
        for r in range(receivers):
            if abs(p[ti,r])< tol:
                p[ti,r] = 0.0

    return p

