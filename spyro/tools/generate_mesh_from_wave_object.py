from firedrake.petsc import PETSc
parprint = PETSc.Sys.Print

def get_domain(wave_object, units):
    import SeismicMesh
    Lz = wave_object.length_z
    Lx = wave_object.length_x

    if units == 'km-s':
        Lz *= 1000
        Lx *= 1000

    bbox = (-Lz, 0.0, 0.0, Lx)
    domain = SeismicMesh.Rectangle(bbox)

    return domain, bbox

def cells_per_wavelength(method, degree, dimension):
    cell_per_wavelength_dictionary = {
        'mlt2tri': 6.70,
        'mlt3tri': 3.55,
        'mlt4tri': 2.41,
        'mlt5tri': 1.84,
        'mlt2tet': 5.85,
        'mlt3tet': 3.08,
    }

    if dimension == 2 and (method == 'MLT' or method == 'CG'):
        cell_type = 'tri'
    if dimension == 3 and (method == 'MLT' or method == 'CG'):
        cell_type = 'tet'

    key = method.lower()+str(degree)+cell_type
    
    return cell_per_wavelength_dictionary.get(key)

def generate_mesh2D(wave_object, pad=0.0, comm = None):
    """ Generates a wave form adapted mesh using parameters from
    the wave object
    """

    import SeismicMesh # ADD TRY EXCEPT

    parprint('Entering mesh generation')

    C = cells_per_wavelength(
        wave_object.method,
        wave_object.degree,
        wave_object.dimension
    )

    domain_pad = pad

    fname = "vel_z6.25m_x12.5m_exact.segy"

    # Bounding box describing domain extents (corner coordinates)
    domain, bbox = get_domain(wave_object)

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

    return model