from firedrake.utility_meshes import UnitSquareMesh
import numpy as np
import firedrake as fire
import spyro
import meshio
import copy
import SeismicMesh
import time


def generate_model(degree):
    method = "KMV"
    quadrature = "KMV"

    pad = 0.75
    Lz = 5.175
    Real_Lz = Lz+ pad
    Lx = 7.5
    Real_Lx = Lx+ 2*pad
    Ly = 7.5
    Real_Ly = Ly + 2*pad

    # time calculations

    final_time = 1.0 

    # receiver calculations

    sources = [(-0.1, 0.5, 0.5)]

    receivers = spyro.create_2d_grid(0.25, 7.25, 0.25, 7.25, 30)
    receivers = spyro.insert_fixed_value(receivers, -0.15, 0)
    
    model = {}
    model["opts"] = {
        "method": 'KMV',
        "variant": None,
        "element": "tetra",  # tria or tetra
        "degree": 3,  # p order
        "dimension": 3,  # dimension
    }
    model["parallelism"] = {
        "type": "automatic",
    }
    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
    }
    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 1,  # damping layer has a exponent variation
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": pad,  # thickness of the PML in the z-direction (km) - always positive
        "lx": pad,  # thickness of the PML in the x-direction (km) - always positive
        "ly": pad,  # thickness of the PML in the y-direction (km) - always positive
    }
    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": len(sources),
        "source_pos": sources,
        "frequency": 5.0,
        "delay": 1.0,
        "num_receivers": len(receivers),
        "receiver_locations": receivers,
    }
    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": final_time,  # Final time for event
        "dt": 0.005,
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 400,  # how frequently to output solution to pvds
        "fspool": 99999,  # how frequently to save solution to RAM
    }
    model['testing_parameters'] = {
            'minimum_mesh_velocity': 1.429,
        }

    return model

def generate_mesh2D_tri(model, comm):

    print('Entering mesh generation', flush = True)
    degree = model['opts']['degree']
    if model['opts']['degree']   == 2:
        M = 7.020
    elif model['opts']['degree'] == 3:
        M = 3.696
    elif model['opts']['degree'] == 4:
        M = 2.664
    elif model['opts']['degree'] == 5:
        M = 2.028

    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx


    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    edge_length = lbda/M

    bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
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

        meshio.write_points_cells("meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".msh",
            points,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".vtk",
            points,[("triangle", cells)],
            file_format="vtk"
            )

    comm.comm.barrier()
    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    print('Finishing mesh generation', flush = True)
    return mesh

def generate_mesh2D_quad(model, comm):
    print('Entering mesh generation', flush = True)
    degree = model['opts']['degree']
    if model['opts']['degree']   == 2:
        M = 4
    elif model['opts']['degree'] == 3:
        M = 4
    elif model['opts']['degree'] == 4:
        M = 4
    elif model['opts']['degree'] == 5:
        M = 4

    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx


    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    edge_length = lbda/M
    nz = int(Real_Lz/edge_length)
    nx = int(Real_Lx/edge_length)

    mesh = fire.RectangleMesh(nx, nz, Real_Lz, Real_Lx, quadrilateral=True)

    coordinates = copy.deepcopy(mesh.coordinates.dat.data)
    mesh.coordinates.dat.data[:,0]=-coordinates[:,0]
    mesh.coordinates.dat.data[:,1]= coordinates[:,1] - lx
    fire.File("meshes/meshQuadTest.pvd").write(mesh)

    return mesh

def generate_mesh3D_tri(model, comm):
    
    
    print('Entering mesh generation', flush = True)
    if model['opts']['degree']   == 2:
        M = 7.020
    elif model['opts']['degree'] == 3:
        M = 3.1

    method = 'KMV'

    pad = 0.75
    Lz = 5.175
    Real_Lz = Lz+ pad
    Lx = 7.5
    Real_Lx = Lx+ 2*pad
    Ly = 7.5
    Real_Ly = Ly + 2*pad

    edge_length = 1.5/(5*M)

    bbox = (-Real_Lz, 0.0,-pad, Lx+pad, -pad, Ly+pad)
    print(bbox)
    print(edge_length)
    cube = SeismicMesh.Cube(bbox)

    # Creating rectangular mesh
    print('Starting seismic mesh', flush = True)
    points, cells = SeismicMesh.generate_mesh(
    domain=cube, 
    edge_length=edge_length, 
    max_iter = 75,
    comm = comm.ensemble_comm,
    verbose = 2
    )

    points, cells = SeismicMesh.sliver_removal(points=points, bbox=bbox, domain=cube, edge_length=edge_length, preserve=True)
    
    print('entering spatial rank 0 after mesh generation')

    meshio.write_points_cells("meshes/hom_overthrust.msh",
        points,[("tetra", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("meshes/hom_overthrust.vtk",
        points,[("tetra", cells)],
        file_format="vtk"
        )

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/hom_overthrust.msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

def generate_mesh(model, comm):
    if   model['opts']['dimension'] == 2:
        return generate_mesh2D(model, comm)
    elif model['opts']['dimension'] == 3:
        return generate_mesh3D(model, comm)

def generate_mesh2D(model, comm):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh2D_tri(model, comm)
    elif model['opts']['method'] == 'CG':
        return generate_mesh2D_quad(model, comm)

def generate_mesh3D(model, comm):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh3D_tri(model, comm)
    elif model['opts']['method'] == 'CG':
        raise ValueError("3D quad mesh not yet implemented")
        #return generate_mesh3D_quad(model, comm)


dimension = 3
method = 'KMV'
if method == 'spectral':
    quadrilateral = True
elif method == 'KMV':
    quadrilateral = False

degree = 3
output = False


model = generate_model(3)
comm = spyro.utils.mpi_init(model)
comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Meshing with overthurst dimensions", flush=True)
comm.comm.barrier()
mesh = generate_mesh(model,comm)


