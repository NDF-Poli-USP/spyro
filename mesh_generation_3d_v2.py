import numpy as np
import firedrake as fire
import meshio
import copy
import SeismicMesh
import spyro

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

def create_model_for_grid_point_calculation(scale):
    
    model = {}
    frequency = 5.0
    minimum_mesh_velocity = 1.429
    lbda = minimum_mesh_velocity/frequency
    pad = scale*lbda
    Lz = scale*15*lbda#100*lbda
    Real_Lz = Lz+ pad
    #print(Real_Lz)
    Lx = scale*30*lbda#90*lbda
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


    receiver_coordinates = create_3d_grid( (bin1_startZ,bin1_startX,bin1_startY)  , (bin1_endZ,bin1_endX,bin1_endY)   , int(np.sqrt(receiver_quantity)))
    # Choose method and parameters
    model["opts"] = {
        "method": 'KMV',
        "variant": None,
        "element": "tetra",  # tria or tetra
        "degree": 3,  # p order
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
        "dt": 0.001,  # timestep size
        "nspool": 200,  # how frequently to output solution to pvds
        "fspool": 100,  # how frequently to save solution to RAM
    }  
    model["parallelism"] = {
    "type": "automatic", 
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model

def wave_solver(model, M, mesh, comm = False):
    minimum_mesh_velocity = 1.429
    method = model["opts"]["method"]
    output = True
    
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(f"Setting up {method} element", flush=True)
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    vp = fire.Constant(minimum_mesh_velocity)

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

def generate_mesh3D(model, M, scale, comm):

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

        meshio.write_points_cells("meshes/homogeneous_3D_scale"+str(int(scale*100))+"by100.msh",
        points,[("tetra", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("meshes/homogeneous_3D_scale"+str(int(scale*100))+"by100.vtk",
        points,[("tetra", cells)],
        file_format="vtk"
        )

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/homogeneous_3D_scale"+str(int(scale*100))+"by100.msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

    print('Finishing mesh generation', flush = True)
    return mesh

mesh_generation = True
# Ms = [2.0, 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
# Ms = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
# Ms = [3.6, 3.7, 3.8, 3.9]
# Ms = [3.9, 4.2, 4.3, 4.4, 4.6]
# Ms = [4.0]
# Ms = [ 4.8, 4.9]
# Ms = [6.45]
# Ms = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
M = 3.6
# scales = [3, 4, 5, 6]
# scales = [0.5, 1.5, 2.5]
scales = [10, 20, 30, 40, 50, 60, 70, 80, 90, 130, 160, 180]

for scale in scales:
    model = create_model_for_grid_point_calculation(scale/100.)
    comm = spyro.utils.mpi_init(model)
    comm.comm.barrier()
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("Meshing with scale div 100="+str(scale), flush=True)
    comm.comm.barrier()
    generate_mesh3D(model, M, scale/100., comm)
