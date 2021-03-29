from mpi4py import MPI
import numpy as np
from scipy import interpolate
import meshio
import SeismicMesh
import firedrake as fire
import time
import copy

import sys
sys.path.append('/home/alexandre/Development/Spyro-main/spyro')
import spyro

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

def create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = 'homogeneous', receiver_type = 'near'):
    ''' Creates models  with the correct parameters for for grid point calculation experiments.
    '''
    model = {}
    # domain calculations
    lbda = minimum_mesh_velocity/frequency
    pml_fraction = lbda
    if receiver_type == 'near':
        Lz = 10*lbda#100*lbda
        Real_Lz = Lz*(1. + 2*pml_fraction)
        Lx = 7*lbda#90*lbda
        Real_Lx = Lx*(1. + 1*pml_fraction)

        # source location
        center_coordinates = [(Real_Lz/2, Real_Lx/2)] #Source at the center. If this is changes receiver's bin has to also be changed.

        # time calculations
        tmin = 1./frequency
        final_time = 10*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 2*lbda#20*lbda
        receiver_bin_width = 2*lbda#15*lbda
        receiver_quantity = 4#2500 # 50 squared

        bin1_startZ = Real_Lz/2. + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endZ   = Real_Lz/2. + receiver_bin_center1 + receiver_bin_width/2.
        bin1_startX = Real_Lx/2. - receiver_bin_width/2.
        bin1_endX   = Real_Lx/2. + receiver_bin_width/2.


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
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
        "exponent": 1,
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": Lz*pml_fraction,  # thickness of the pml in the z-direction (km) - always positive
        "lx": Lx*pml_fraction,  # thickness of the pml in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "demos/mm_exact.msh",
        "initmodel": "demos/mm_init.hdf5",
        "truemodel": "demos/mm_exact.hdf5",
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": center_coordinates,
        "frequency": frequency,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity)))
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
        'experiment_type': 'homogeneous',
        'minimum_mesh_velocity': minimum_mesh_velocity,
        'pml_fraction': pml_fraction,
        'receiver_type': receiver_type,
        'source_sigma': 500.0
    }

    return model

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
        #creating disk around source
        disk_M = grid_point_to_mesh_point_converter_for_seismicmesh(model, 15)
        disk = SeismicMesh.Disk([Real_Lz/2, Real_Lx/2], lbda)
        disk_points, disk_cells = SeismicMesh.generate_mesh(
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
        pfix = disk_points,
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

    print('Finishing mesh generation', flush = True)
    return mesh

def wave_solver(model, G, comm = False):
    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']

    mesh = generate_mesh(model, G, comm)
    
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = fire.FunctionSpace(mesh, element)

    vp_exact = fire.Constant(minimum_mesh_velocity)

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

print("Starting SSPRK wave solver", flush = True)

frequency = 5.0
degree = 1
minimum_mesh_velocity = 1.0
method = 'CG'

model = create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = 'homogenous', receiver_type = 'near')
#print("Model built at time "+str(time.time()-start_time), flush = True)
comm = spyro.utils.mpi_init(model)
#print("Comm built at time "+str(time.time()-start_time), flush = True)

p_exact = wave_solver(model, G =9, comm = comm)
print("p_exact finished at time "+str(time.time()-start_time), flush = True)