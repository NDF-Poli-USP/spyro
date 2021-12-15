import spyro
from spyro.utils.synthetic import smooth_field
from SeismicMesh import write_velocity_model
import SeismicMesh
import meshio
import copy
import firedrake as fire
import os

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

    #bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
    bbox = (-Lz, 0.0, 0.0, Lx)
    domain = SeismicMesh.Rectangle(bbox)

    return domain, bbox

def build_mesh(model, frequency, output_filename, vp_filename, units = 'km-s', see_mesh = True):
    dimension = 2
    method = 'KMV'
    degree = 4
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

    mesh = fire.Mesh(
            meshfname,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    
    return mesh


# try 100:
frequency = 2

input_vp_model = "velocity_models/vp_marmousi-ii_smoother_guess.segy"
output_mesh_name = "meshes/fwi_mesh_"+str(frequency)+"Hz.msh"


model = {}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/fwi_mesh_2Hz.msh",
    "initmodel": "velocity_models/vp_marmousi-ii_smoother_guess.hdf5",
    "truemodel": "not_used.hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}


vp_filename, vp_filetype = os.path.splitext(input_vp_model)

write_velocity_model(input_vp_model, ofname = vp_filename)
new_vpfile = vp_filename+'.hdf5'

print('Entering mesh generation', flush = True)

mesh_filename = output_mesh_name

mesh = build_mesh(model, frequency, mesh_filename, vp_filename+'.segy' )