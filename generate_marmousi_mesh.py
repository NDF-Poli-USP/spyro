import spyro
from SeismicMesh import write_velocity_model
import SeismicMesh
import meshio
import copy
import firedrake as fire
import os
import sys

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

## Para gerar a malha com dimensoes Lz e Lx em km
# try 100:
Lz = 3.5
Lx = 17.0
frequency = float(sys.argv[1])
degree = int(sys.argv[2])
pad_size = float(sys.argv[3])

input_vp_model = "/media/olender/Extreme SSD/common_files/velocity_models/vp_marmousi-ii.segy"
output_mesh_name = "meshes/marmousi_f"+str(frequency)+"_degree"+str(degree)+"_pad"+str(pad_size)  #Onde vamos salvar a malha
show = True #Gera um .pvd para visualizar no paraview o modelo de velocidade
pad = pad_size

vp_filename, vp_filetype = os.path.splitext(input_vp_model)
if vp_filetype == ".segy":
    input_vp_model = vp_filename+".segy"
    write_velocity_model(input_vp_model, ofname = vp_filename)
    new_vpfile = vp_filename+'.hdf5'
elif vp_filetype == ".hdf5":
    new_vpfile = vp_filename+'.hdf5'


model = {}
model['inversion'] = {
    'true_model': vp_filename+".hdf5",
}
model["mesh"] = {
    "Lz": Lz,  # depth in km - always positive
    "Lx": Lx,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": None,
    "initmodel": vp_filename+".hdf5",
    "truemodel": vp_filename+".hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": pad,  # thickness of the PML in the z-direction (km) - always positive
    "lx": pad,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

print('Entering mesh generation', flush = True)

mesh_filename = output_mesh_name

mesh = build_mesh(model, frequency, mesh_filename, vp_filename+'.segy' )

if show == True:
    V = fire.FunctionSpace(mesh, 'KMV', degree)
    vp = spyro.io.interpolate(model, mesh, V, guess=False)
    output = fire.File('velocity_starting_for_'+str(frequency)+'.pvd')
    output.write(vp)




