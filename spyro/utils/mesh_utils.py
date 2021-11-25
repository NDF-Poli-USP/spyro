import numpy as np
from scipy.interpolate import griddata
import copy
import meshio
import SeismicMesh
import firedrake as fire
import spyro

def cells_per_wavelength(method, degree, dimension):
    cell_per_wavelength_dictionary = {
        'kmv2tri': 7.02,
        'kmv3tri': 3.70,
        'kmv4tri': 2.67,
        'kmv5tri': 2.03,
        'kmv2tet': 6.12,
        'kmv3tet': 3.72,
    }

    if dimension == 2 and (method == 'KMV' or method == 'CG'):
        cell_type = 'tri'
    if dimension == 3 and (method == 'KMV' or method == 'CG'):
        cell_type = 'tet'

    key = method.lower()+str(degree)+cell_type
    
    return cell_per_wavelength_dictionary.get(key)

def get_domain(model, units):
    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']
    if units == 'km-s':
        Lz *= 1000
        lz *= 1000
        Lx *= 1000
        lx *= 1000

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    if model["opts"]["dimension"] == 2:
        #bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
        bbox = (-Lz, 0.0, 0.0, Lx)
        domain = SeismicMesh.Rectangle(bbox)
    elif model["opts"]["dimension"] == 3:
        Ly = model["mesh"]['Ly']
        ly= model['BCs']['ly']
        Real_Ly = Ly + 2*ly

        bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx, -ly, Real_Ly-ly)
        domain = SeismicMesh.Cube(bbox)

    return domain, bbox

def build_mesh(model, comm, output_filename, vp_filename, units = 'km-s', see_mesh = True):
    dimension = model["opts"]["dimension"]
    method =model["opts"]["method"]
    degree =model["opts"]["degree"]
    frequency = model["acquisition"]['frequency']
    domain_pad = model["BCs"]["lz"]

    if units == 'km-s':
        minimum_mesh_velocity = 1.429
    else:
        minimum_mesh_velocity = 1429

    C = cells_per_wavelength(method, degree, dimension)
    hmin = minimum_mesh_velocity/(C*frequency)

    domain, bbox = get_domain(model, units = units)
    
    if units == 'km-s':
        hmin *= 1000
        domain_pad *= 1000


    ef = SeismicMesh.get_sizing_function_from_segy(
        vp_filename,
        bbox,
        hmin=hmin,
        wl=C,
        freq=frequency,
        grade=0.15,
        domain_pad=domain_pad,
        pad_style="edge",
        units = units,
    )

    if comm.comm.rank == 0:
        # Creating rectangular mesh
        points, cells = SeismicMesh.generate_mesh(
            domain=domain, 
            edge_length=ef, 
            verbose = 0, 
            mesh_improvement=False 
            )

        print('entering spatial rank 0 after mesh generation')

        meshio.write_points_cells(output_filename+".msh",
            points/ 1000,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
        )
        
        meshfname = copy.deepcopy(output_filename+".msh")
        
        if see_mesh == True:
            meshio.write_points_cells(output_filename+".vtk",
                points/ 1000,[("triangle", cells)],
                file_format="vtk"
            )

    comm.comm.barrier()
    mesh = fire.Mesh(
            meshfname,
            comm=comm.comm,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    
    return mesh

def remesh(model, vp, V, output_mesh_filename, output_vp_filename, comm):
    # Interpolate vp to a structured grid
    grid_spacing = 10.0 / 1000.0
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coordinates = fire.interpolate(m.coordinates, W)
    x, y = coordinates.dat.data[:, 0], coordinates.dat.data[:, 1]

    # add buffer to avoid NaN when calling griddata
    min_x = np.amin(x) + 0.01
    max_x = np.amax(x) - 0.01
    min_y = np.amin(y) + 0.01
    max_y = np.amax(y) - 0.01

    z = function.dat.data[:] * 1000.0  # convert from km/s to m/s

    # target grid to interpolate to
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    v_map = griddata((x, y), z, (xi, yi), method="linear")
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("creating new velocity model...", flush=True)
        spyro.io.create_segy(v_map, output_vp_filename)

    mesh = build_mesh(model, comm, output_mesh_filename, output_vp_filename)

    return mesh

