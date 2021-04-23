import firedrake as fire
import numpy as np
from mpi4py import MPI
import meshio
import SeismicMesh
import time

import sys
sys.path.append('/home/alexandre/Development/Spyro-new_source')
import spyro

def create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity= 1.0, experiment_type = 'homogeneous', receiver_type = 'near'):
    ''' Creates models  with the correct parameters for for grid point calculation experiments.
    
    Parameters
    ----------
    frequency: `float`
        Source frequency to use in calculation
    degree: `int`
        Polynomial degree of finite element space
    method: `string`
        The finite element method choosen
    minimum_mesh_velocity: `float`
        Minimum velocity presented in the medium
    experiment_type: `string`
        Only options are `homogenous` or `heterogenous`
    receiver_type: `string`
        Options: `near`, `far` or `near_and_far`. Specifies receiver grid locations for experiment

    Returns
    -------
    model: Python `dictionary`
        Contains model options and parameters for use in Spyro
        

    '''
    model = {}
    # domain calculations
    if experiment_type == 'homogeneous':
        lbda = minimum_mesh_velocity/frequency
        pml_fraction = lbda
        Lz = 30*lbda#100*lbda
        Real_Lz = Lz*(1. + pml_fraction)
        Lx = 20*lbda#90*lbda
        Real_Lx = Lx*(1. + 2*pml_fraction)

        # source location
        source_coordinates = [(Real_Lz/2, Real_Lx/2)] #Source at the center. If this is changes receiver's bin has to also be changed.
        source_z = Real_Lz/2.
        source_x = Real_Lx/2.
        padz = Lz*pml_fraction
        padx = Lx*pml_fraction

    if experiment_type == 'heterogenous':
        #using the BP2004 velocity model
        minimum_mesh_velocity = 1429.0/1000.0
        Lz = 12000.0/1000.
        Lx = 67000.0/1000.
        pad = 1000./1000.
        Real_Lz = Lz+ pad
        Real_Lx = Lx+ 2*pad
        source_z = -1.0
        source_x = Real_Lx/2.
        source_coordinates = [(source_z,source_x)]
        SeismicMesh.write_velocity_model('vel_z6.25m_x12.5m_exact.segy', ofname = 'velocity_models/bp2004')
        padz = pad
        padx = pad
    
    if receiver_type == 'near' and experiment_type == 'homogeneous':

        # time calculations
        tmin = 1./frequency
        final_time = 35*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 5*lbda#20*lbda
        receiver_bin_width = 5*lbda#15*lbda
        receiver_quantity = 16#2500 # 50 squared

        bin1_startZ = source_z + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endZ   = source_z + receiver_bin_center1 + receiver_bin_width/2.
        bin1_startX = source_x - receiver_bin_width/2.
        bin1_endX   = source_x + receiver_bin_width/2.
        receiver_coordinates = spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity)))

    if receiver_type == 'near' and experiment_type == 'heterogenous':

        # time calculations
        tmin = 1./frequency
        final_time = 2*10*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 2.5*750.0/1000
        receiver_bin_width = 500.0/1000
        receiver_quantity_in_bin = 100#2500 # 50 squared

        bin1_startZ = source_z - receiver_bin_width/2.
        bin1_endZ   = source_z + receiver_bin_width/2.
        bin1_startX = source_x + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endX   = source_x + receiver_bin_center1 + receiver_bin_width/2.

        receiver_coordinates = spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity_in_bin)))

        receiver_bin_center2 = 6500.0/1000
        receiver_bin_width = 500.0/1000

        bin2_startZ = source_z - receiver_bin_width/2.
        bin2_endZ   = source_z + receiver_bin_width/2.
        bin2_startX = source_x + receiver_bin_center2 - receiver_bin_width/2.
        bin2_endX   = source_x + receiver_bin_center2 + receiver_bin_width/2.

        receiver_coordinates= receiver_coordinates + spyro.create_2d_grid(bin2_startZ, bin2_endZ, bin2_startX, bin2_endX, int(np.sqrt(receiver_quantity_in_bin))) 

        receiver_quantity = 2*receiver_quantity_in_bin

    elif receiver_type == 'far':
        raise ValueError('Far receivers minimum grid point calculation experiment not implemented because of computational limits.')
    
    # Choose method and parameters
    model["opts"] = {
        "method": method,
        "variant": None,
        "element": "tria",  # tria or tetra
        "degree": degree,  # p order
        "dimension": 2,  # dimension
    }

    model["PML"] = {
        "status": False,  # True or false
        "outer_bc": False, #"non-reflective",  #  neumann, non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
        "exponent": 1,
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": padz,  # thickness of the pml in the z-direction (km) - always positive
        "lx": padx,  # thickness of the pml in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "demos/mm_exact.msh",
        "initmodel": "velocity_models/bp2004.hdf5",
        "truemodel": "velocity_models/bp2004.hdf5",
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": source_coordinates,
        "source_mesh_point": False,
        "source_point_dof": False,
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
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
    }
    model['testing_parameters'] = {
        'experiment_type': experiment_type,
        'minimum_mesh_velocity': minimum_mesh_velocity,
        'pml_fraction': padz/Lz,
        'receiver_type': receiver_type
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model

def generate_mesh(model,G, comm):
    
    print('Entering mesh generation', flush = True)
    M = spyro.tools.grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    disk_M = spyro.tools.grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
    method = model["opts"]["method"]
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

    bbox = (0.0, Real_Lz, 0.0, Real_Lx)
    rec = SeismicMesh.Rectangle(bbox)

    if comm.comm.rank == 0:
        #creating disk around source
        disk_M = spyro.tools.grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
        disk = SeismicMesh.Disk([Real_Lz/2, Real_Lx/2], lbda)
        disk_points, disk_cells = SeismicMesh.generate_mesh(
            domain=disk,
            edge_length=lbda/disk_M,
            mesh_improvement = False,
            comm = comm.ensemble_comm,
            verbose = 0
        )
        meshio.write_points_cells("meshes/disk"+str(G)+".vtk",
            disk_points,[("triangle", disk_cells)],
            file_format="vtk"
            )

        if model['acquisition']['source_mesh_point']:
            source_position = model['acquisition']['source_pos']
            fixed_points = np.append(disk_points,source_position, axis=0)
        else:
            fixed_points = disk_points

        # Creating rectangular mesh
        points, cells = SeismicMesh.generate_mesh(
        domain=rec, 
        edge_length=edge_length, 
        mesh_improvement = False,
        comm = comm.ensemble_comm,
        pfix = fixed_points ,
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

    if model['acquisition']['source_mesh_point']== True:
        element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
        V = fire.FunctionSpace(mesh, element)


    print('Finishing mesh generation', flush = True)

    return mesh

def generate_mesh_immersed_disk(model,G, comm):
    
    print('Entering mesh generation', flush = True)
    M = spyro.tools.grid_point_to_mesh_point_converter_for_seismicmesh(model, G)
    disk_M = spyro.tools.grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
    method = model["opts"]["method"]
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

    bbox = (0.0, Real_Lz, 0.0, Real_Lx)
    disk = SeismicMesh.Disk([Real_Lz/2, Real_Lx/2], lbda)
    rec = SeismicMesh.Rectangle(bbox)

    if comm.comm.rank == 0:
        #creating disk around source
        
        if model['acquisition']['source_mesh_point']:
            source_position = model['acquisition']['source_pos']
            fixed_points = np.append(disk_points,source_position, axis=0)
        else:
            fixed_points = None

        # Creating rectangular mesh
        points, cells = SeismicMesh.generate_mesh(
        domain=rec, 
        edge_length=edge_length, 
        mesh_improvement = False,
        comm = comm.ensemble_comm,
        subdomains = [disk] ,
        pfix = fixed_points ,
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
        meshio.write_points_cells("meshes/IMMERSEDhomogeneous"+str(G)+".vtk",
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

    if model['acquisition']['source_mesh_point']== True:
        element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
        V = fire.FunctionSpace(mesh, element)


    print('Finishing mesh generation', flush = True)

    return mesh

def wave_solver(model, G, comm = False):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']

    mesh = generate_mesh_immersed_disk(model, G, comm)
    
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

    #sources = spyro.Sources(model, mesh, V, comm).create()
    sources = spyro.Updated_sources(model, mesh, V, comm).create()
    receivers = spyro.Receivers(model, mesh, V, comm).create()

    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            t1 = time.time()
            p_field, p_recv = spyro.solvers.Old_Leapfrog(
                model, mesh, comm, vp_exact, sources, receivers, source_num=sn, output= True, G = G
            )
            print(time.time() - t1)

    return p_recv

frequency = 5.0
degree = 2
method = 'KMV'
G=9.0

model = create_model_for_grid_point_calculation(frequency,degree,method)

# Create the computational environment
comm = spyro.utils.mpi_init(model)

p_rec = spyro.tools.wave_solver(model,G,comm=comm)