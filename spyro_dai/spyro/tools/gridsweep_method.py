from datetime import datetime
from mpi4py import MPI
import firedrake as fire
import pickle
import spyro

def saving_source_and_receiver_location_in_csv(model):
    file_name = 'experiment/sources.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for source in model['acquisition']['source_pos']:
        z, x = source
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    file_name = 'experiment/receivers.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for receiver in model['acquisition']['receiver_locations']:
        z, x = receiver
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    return None

def save_pickle(filename, array):
    """Save a `numpy.ndarray` to a `pickle`.

    Parameters
    ----------
    filename: str
        The filename to save the data as a `pickle`
    array: `numpy.ndarray`
        The data to save a pickle (e.g., a shot)

    Returns
    -------
    None

    """
    with open(filename, "wb") as f:
        pickle.dump(array, f)
    return None

def load_pickle(filename):
    with open(filename, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
    return array

def wave_solver(model, mesh, comm, output = False):
    method = model["opts"]["method"]
    degree = model['opts']['degree']

    element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")
    V = fire.FunctionSpace(mesh, element)
    vp = fire.Constant(1.429)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Finding sources and receivers", flush=True)
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Starting simulation", flush=True)

    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )
    p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output = output)
    return p_r

def read_mesh(M):

    mesh = fire.Mesh(
        "meshes/homogeneous_3D_M"+str(M)+".msh",
        distribution_parameters={
            "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
        },
    )
    return mesh

def estimate_timestep(mesh, V, c, estimate_max_eigenvalue=True):
    """Estimate the maximum stable timestep based on the spectral radius
    using optionally the Gershgorin Circle Theorem to estimate the
    maximum generalized eigenvalue. Otherwise computes the maximum
    generalized eigenvalue exactly

    ONLY WORKS WITH KMV ELEMENTS
    """

    u, v = fire.TrialFunction(V), fire.TestFunction(V)
    quad_rule = finat.quadrature.make_quadrature(
        V.finat_element.cell, V.ufl_element().degree(), "KMV"
    )
    dxlump = fire.dx(rule=quad_rule)
    A = fire.assemble(u * v * dxlump)
    ai, aj, av = A.petscmat.getValuesCSR()
    av_inv = []
    for value in av:
        if value == 0:
            av_inv.append(0.0)
        else:
            av_inv.append(1 / value)
    Asp = scipy.sparse.csr_matrix((av, aj, ai))
    Asp_inv = scipy.sparse.csr_matrix((av_inv, aj, ai))

    K = fire.assemble(c*c*dot(grad(u), grad(v)) * dxlump)
    ai, aj, av = K.petscmat.getValuesCSR()
    Ksp = scipy.sparse.csr_matrix((av, aj, ai))

    # operator
    Lsp = Asp_inv.multiply(Ksp)
    if estimate_max_eigenvalue:
        # absolute maximum of diagonals
        max_eigval = np.amax(np.abs(Lsp.diagonal()))
    else:
        print(
            "Computing exact eigenvalues is extremely computationally demanding!",
            flush=True,
        )
        max_eigval = scipy.sparse.linalg.eigs(
            Ksp, M=Asp, k=1, which="LM", return_eigenvectors=False
        )[0]

    # print(max_eigval)
    if np.sqrt(max_eigval) > 0.0:
    	max_dt = np.float(2 / np.sqrt(max_eigval))
    else:
        max_dt = 100000000
    #print(
    #    f"Maximum stable timestep should be about: {np.float(2 / np.sqrt(max_eigval))} seconds",
    #    flush=True,
    #)
    return max_dt

def generate_mesh3D(model, M, comm):

    print('Entering mesh generation', flush = True)
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

    minimum_mesh_velocity = 1.429
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    edge_length = lbda/M

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

        meshio.write_points_cells("meshes/homogeneous_3D_M"+str(M)+".msh",
        points,[("tetra", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("meshes/homogeneous_3D_M"+str(M)+".vtk",
        points,[("tetra", cells)],
        file_format="vtk"
        )

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/homogeneous_3D_M"+str(M)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

    print('Finishing mesh generation', flush = True)
    return mesh

def dt_to_use(degree, M):
    if degree == 3:
        if   M == 2.1:
            dt = 0.000667
        elif M == 2.4:
            dt = 0.000632
        elif M == 2.6:
            dt = 0.000516
        elif M == 2.7:
            dt = 0.000408
        elif M == 2.9:
            dt = 0.000425
        elif M == 3.1:
            dt = 0.00110
        elif M == 3.3:
            dt = 0.00103
        elif M == 3.6:
            dt = 0.000939
        else:
            raise ValueError('dt not yet calculated')
    
    return dt

def create_3d_grid(start, end, num):
    """Create a 3d grid of `num**3` points between `start1`
    and `end1` and `start2` and `end2`

    Parameters
    ----------
    start: tuple of floats
        starting position coordinate
    end: tuple of floats
        ending position coordinate
    num: integer
        number of receivers between `start` and `end`

    Returns
    -------
    receiver_locations: a list of tuples

    """
    (start1, start2, start3) = start
    (end1, end2, end3)  = end
    x = np.linspace(start1, end1, num)
    y = np.linspace(start2, end2, num)
    z = np.linspace(start3, end3, num)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return [tuple(point) for point in points]

def create_model_for_grid_point_calculation3D(degree):
    
    model = {}
    frequency = 5.0
    minimum_mesh_velocity = 1.429
    lbda = minimum_mesh_velocity/frequency
    pad = lbda
    Lz = 15*lbda#100*lbda
    Real_Lz = Lz+ pad
    #print(Real_Lz)
    Lx = 30*lbda#90*lbda
    Real_Lx = Lx+ 2*pad
    Ly = Lx
    Real_Ly = Ly + 2*pad

    # source location
    source_z = -Real_Lz/2.#1.0
    #print(source_z)
    source_x = lbda*1.5
    source_y = Real_Ly/2.0
    source_coordinates = [(source_z, source_x, source_y)] #Source at the center. If this is changes receiver's bin has to also be changed.
    padz = pad
    padx = pad
    pady = pad

    # time calculations
    tmin = 1./frequency
    final_time = 20*tmin #should be 35

    # receiver calculations

    receiver_bin_center1 = 10*lbda#20*lbda
    receiver_bin_width = 5*lbda#15*lbda
    receiver_quantity = 36#2500 # 50 squared

    bin1_startZ = source_z - receiver_bin_width/2.
    bin1_endZ   = source_z + receiver_bin_width/2.
    bin1_startX = source_x + receiver_bin_center1 - receiver_bin_width/2.
    bin1_endX   = source_x + receiver_bin_center1 + receiver_bin_width/2.
    bin1_startY = source_y - receiver_bin_width/2.
    bin1_endY   = source_y + receiver_bin_width/2.

    receiver_coordinates = create_3d_grid( (bin1_startZ,bin1_startX,bin1_startY)  , (bin1_endZ,bin1_endX,bin1_endY)   , 6)
    # Choose method and parameters
    model["opts"] = {
        "method": 'KMV',
        "variant": None,
        "element": "tetra",  # tria or tetra
        'quadrature': 'KMV',
        "degree": degree,  # p order
        "dimension": 3,  # dimension
    }

    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
        "exponent": 1,
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": padz,  # thickness of the pml in the z-direction (km) - always positive
        "lx": padx,  # thickness of the pml in the x-direction (km) - always positive
        "ly": pady,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": source_coordinates,
        "frequency": frequency,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": receiver_coordinates,
    }

    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": final_time,  # Final time for event
        "dt": 0.0002,  # timestep size
        "nspool": 200,  # how frequently to output solution to pvds
        "fspool": 100,  # how frequently to save solution to RAM
    }  
    model["parallelism"] = {
    "type": "spatial", 
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model

def executing_gridsweep(sweep_parameters):
    
    # IO parameters
    mesh_generation   = sweep_parameters['IO']['generate_meshes']
    saving_results    = sweep_parameters['IO']['output_pickle_of_wave_propagator_results']
    loading_results   = sweep_parameters['IO']['load_existing_pickle_of_wave_propagator_results']
    loading_reference = sweep_parameters['IO']['use_precomputed_reference_case']
    filenameprefix    = sweep_parameters['IO']['grid_sweep_data_filename_prefix']
    calculate_dt      = sweep_parameters['IO']['calculate_maximum_timestep']
    save_receiver_location = sweep_parameters['IO']['save_receiver_location']

    # Reference case for error comparison
    G_reference = sweep_parameters['reference']['G']
    P_reference = sweep_parameters['reference']['P']

    # Degrees and Gs for sweep
    Gs      = sweep_parameters['sweep_list']['DoFs']
    degrees = sweep_parameters['sweep_list']['Ps']

    # Experiment parameters
    experiment_type = sweep_parameters['experiment']['velocity_type']
    method          = sweep_parameters['experiment']['method']
    frequency       = sweep_parameters['experiment']['frequency']
    receiver_type   = sweep_parameters['experiment']['receiver_disposition']
    if experiment_type == 'homogeneous':
        minimum_mesh_velocity = sweep_parameters['experiment']['minimum_mesh_velocity']
    elif experiment_type == 'heterogeneous':
        minimum_mesh_velocity = False

    ## Generating comm
    model = spyro.tools.create_model_for_grid_point_calculation(frequency,1,method,minimum_mesh_velocity,experiment_type=experiment_type, receiver_type = receiver_type)
    comm = spyro.utils.mpi_init(model)

    ## Output file for saving data
    date = datetime.today().strftime('%Y_%m_%d')
    filename = filenameprefix+date
    text_file = open(filename+".txt", "w")
    text_file.write(experiment_type+' and '+method+' \n')

    ## Generating csv file for visualizing receiver and source position in paraview
    if save_receiver_location == True:
        saving_source_and_receiver_location_in_csv(model)

    if loading_reference == False:
        model = spyro.tools.create_model_for_grid_point_calculation(frequency, P_reference, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = receiver_type)
        print('Calculating reference solution of G = '+str(G_reference)+' and p = '+str(P_reference), flush = True)
        p_exact = spyro.tools.wave_solver(model, G =G_reference, comm = comm)
        save_pickle("experiment/"+filenameprefix+"_heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+".pck", p_exact)
    else:
        p_exact = load_pickle("experiment/"+filenameprefix+"_heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+".pck")

    ## Starting sweep
    for degree in degrees:
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print('\nFor p of '+str(degree), flush = True)
            text_file.write('For p of '+str(degree)+'\n')
            print('Starting sweep:', flush = True)

        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            text_file.write('\tG\t\tError \n')
        for G in Gs:
            model = create_model_for_grid_point_calculation3D(degree)
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print('G of '+str(G), flush = True)

            comm.comm.barrier()
            if mesh_generation == True:
                mesh = generate_mesh3D(model, G, comm)
            else:
                mesh = read_mesh(G)
            comm.comm.barrier()

            if calculate_dt == True:
                method = model["opts"]["method"]
                degree = model['opts']['degree']
                element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")
                V = fire.FunctionSpace(mesh, element)
                vp = fire.Constant(1.429)
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    print("Calculating new dt", flush=True)
                new_dt = 0.2*estimate_timestep(mesh, V, vp,estimate_max_eigenvalue=True)
                comm.comm.barrier()
                model['timeaxis']['dt'] = comm.comm.allreduce(new_dt, op=MPI.MIN)
            else:
                model['timeaxis']['dt'] = dt_to_use(degree, G)
                comm.comm.barrier()

            if loading_results == True:
                p_0 = load_pickle("experiment/"+filenameprefix+"_heterogeneous_p"+str(P_reference)+"g"+str(G_reference)+".pck")
            elif loading_results == False:
                p_0 = wave_solver(model, mesh, G =G, comm = comm)
            
            if saving_results == True:
                save_pickle("experiment/"+filenameprefix+"_heterogeneous_p"+str(P_reference)+"g"+str(G_reference)+".pck", p_0)

            comm.comm.barrier()
            error = spyro.tools.error_calc(p_exact, p_0, model, comm = comm)
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)
                text_file.write('\t'+ str(G) +'\t\t'+str(error)+' \n')

    text_file.close()
