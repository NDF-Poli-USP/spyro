from firedrake.utility_meshes import UnitSquareMesh
import numpy as np
import firedrake as fire
import spyro
import meshio
import copy
import SeismicMesh
import time

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

def generate_model(degree, int_scale):
    method = "KMV"
    quadrature = "KMV"
    scale = int_scale/100.

    lbda = 1.429/5.0
    pad = lbda
    Lz = 8.
    Real_Lz = Lz+ pad
    Lx = 8.
    Real_Lx = Lx+ 2*pad
    Ly = 8.
    Real_Ly = Ly + 2*pad
    pad *= scale
    Real_Lz *= scale
    Real_Lx *= scale
    Real_Ly *= scale


    # source location
    source_z = -Real_Lz/2.#1.0
    #print(source_z)
    source_x = lbda*1.5
    source_y = Real_Ly/2.0
    source_coordinates = [(source_z, source_x, source_y)] #Source at the center. If this is changes receiver's bin has to also be changed.

    # time calculations

    final_time = 1.0 #should be 35

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
        "num_sources": 1,
        "source_pos": source_coordinates,
        "frequency": 5.0,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": receiver_coordinates,
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

def generate_mesh3D_tri(model, comm, scale):
    
    print('Entering mesh generation', flush = True)
    if model['opts']['degree']   == 2:
        M = 7.020
    elif model['opts']['degree'] == 3:
        M = 3.1

    method = 'KMV'
    lbda = 1.429/5.0
    pad = lbda

    Lz = 8.
    Lx = 8.
    Ly = 8.
    Real_Lz = Lz + pad
    Real_Lx = Lx + 2*pad
    Real_Ly = Ly + 2*pad

    pad *= scale
    Real_Lz *= scale
    Real_Lx *= scale
    Real_Ly *= scale

    edge_length = pad/M

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
    
   # points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
   # a=np.amin(SeismicMesh.geometry.simp_qual(points, cells))

    meshio.write_points_cells("meshes/ico_3D_scale"+str(int(scale*100))+"by100.msh",
        points,[("tetra", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("meshes/ico_3D_scale"+str(int(scale*100))+"by100.vtk",
        points,[("tetra", cells)],
        file_format="vtk"
        )

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/ico_3D_scale"+str(int(scale*100))+"by100.msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )

def generate_mesh(model, comm, scale):
    scale = scale / 100.
    if   model['opts']['dimension'] == 2:
        return generate_mesh2D(model, comm)
    elif model['opts']['dimension'] == 3:
        return generate_mesh3D(model, comm, scale)

def generate_mesh2D(model, comm):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh2D_tri(model, comm)
    elif model['opts']['method'] == 'CG':
        return generate_mesh2D_quad(model, comm)

def generate_mesh3D(model, comm, scale):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh3D_tri(model, comm, scale)
    elif model['opts']['method'] == 'CG':
        raise ValueError("3D quad mesh not yet implemented")
        #return generate_mesh3D_quad(model, comm)



dimension = 3
method = 'KMV'
if method == 'spectral':
    quadrilateral = True
elif method == 'KMV':
    quadrilateral = False

degrees = [3]
output = False

scales = [10, 50, 100]

for scale in scales:
    model = generate_model(3, scale)
    comm = spyro.utils.mpi_init(model)
    comm.comm.barrier()
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("Meshing with scale div 100="+str(scale), flush=True)
    comm.comm.barrier()
    mesh = generate_mesh(model,comm, scale)


