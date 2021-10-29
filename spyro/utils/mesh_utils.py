import numpy as np
import copy
import meshio
import SeismicMesh
import firedrake as fire

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

    key = method.lowercase+str(degree)+cell_type
    
    return cell_per_wavelength_dictionary.get(key)

def get_domain(model):
    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx

    if model["opts"]["dimension"] == 2:
        bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
        domain = SeismicMesh.Rectangle(bbox)
    elif model["opts"]["dimension"] == 3:
        Ly = model["mesh"]['Ly']
        ly= model['BCs']['ly']
        Real_Ly = Ly + 2*ly

        bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx, -ly, Real_Ly-ly)
        domain = SeismicMesh.Cube(bbox)

    return domain, bbox

def build_mesh(model, comm, fname, vp = None, units = 'km-s', see_mesh = True):
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

    domain, bbox = get_domain(model)
    
    if units == 'km-s':
        hmin *= 1000
        domain_pad *= 1000

    if vp == None:
        ef = SeismicMesh.get_sizing_function_from_segy(
            fname,
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

            meshio.write_points_cells(fname,
                points/ 1000,[("triangle", cells)],
                file_format="gmsh22", 
                binary = False
            )
            
            meshfname = copy.deepcopy(fname)
            
            if see_mesh == True:
                fname =fname[:,-4]
                meshio.write_points_cells(fname+".vtk",
                    points/ 1000,[("triangle", cells)],
                    file_format="vtk"
                )
    
    mesh = fire.Mesh(
            meshfname,
            comm=comm,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    
    return mesh


