from pickle import TRUE
import string
from fwi_aux import forward_solver, fwi_solver
import spyro
from spyro.utils.synthetic import smooth_field
from SeismicMesh import write_velocity_model
import SeismicMesh
from pyop2.mpi import COMM_WORLD
from mpi4py import MPI
import meshio
import copy
import firedrake as fire
import os

def get_file_type(str):
    strings = str.split('.')
    if len(strings) >= 1:
        if strings[-1]=='hdf5' or strings[-1]=='segy':
            file_type = '.'+strings[-1]
            num_to_remove = len(file_type)
            file_name = str[:-num_to_remove]
        else:
            file_type = None
            file_name = str
    
    return file_name, file_type

def create_model():
    num_sources = 10

    model = {}
    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadratrue": "KMV",  # Equi or KMV
        "degree": 4,  # p order
        "dimension": 2,  # dimension
        "regularization": True,  # regularization is on?
        "gamma": 1.0e-6,  # regularization parameter
    }
    model["parallelism"] = {
        "type": "automatic",
    }
    model["inversion"] = {
        "initial_guess" : None,
        "true_model" : None,
        "regularization" : True,
        "gamma" : 1e-6,
        "gradient_smoothing" : False,
        "gamma2": None,
        "shot_record" : None,
    }
    model["mesh"] = {
        "Lz": 3.5,  # depth in km - always positive
        "Lx": 17.0/2.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": None,
        "initmodel": None,
        "truemodel": None,
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
    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": num_sources,
        "source_pos": spyro.create_transect((-0.01, 0.5), (-0.01, 8.0), num_sources),
        "frequency": None,
        "delay": 1.0,
        "num_receivers": 500,
        "receiver_locations": spyro.create_transect((-0.01, 0.1), (-0.01, 8.4), 500),
    }
    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": 1.00,  # Final time for event
        "dt": 0.001,
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 100,  # how frequently to output solution to pvds
        "fspool": 10,  # how frequently to save solution to RAM
        "skip": 1,
    }

    return model

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

def build_mesh(model, frequency, output_filename, vp_filename, comm, units = 'km-s', see_mesh = True):
    dimension = 2
    method = 'KMV'
    degree = 4
    domain_pad = 0.05*model["BCs"]["lz"]

    if units == 'km-s':
        minimum_mesh_velocity = 1.429
    else:
        minimum_mesh_velocity = 1429

    C = cells_per_wavelength(method, degree, dimension)
    hmin = minimum_mesh_velocity/(C*frequency)

    meshfname = copy.deepcopy(output_filename+".msh")

    if comm.ensemble_comm.rank ==0:

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
            comm=comm.comm
        )

        points, cells = SeismicMesh.generate_mesh(
            domain=domain, 
            edge_length=ef, 
            verbose = 0, 
            mesh_improvement=False,
            comm=comm.comm,
            )

        s_print('entering spatial rank 0 after mesh generation',comm )
        if comm.comm.rank == 0:

            meshio.write_points_cells(output_filename+".msh",
                points/ 1000,[("triangle", cells)],
                file_format="gmsh22", 
                binary = False
            )
            
            if see_mesh == True:
                meshio.write_points_cells(output_filename+".vtk",
                    points/ 1000,[("triangle", cells)],
                    file_format="vtk"
                )

    comm.comm.barrier()
    comm.ensemble_comm.barrier()
    mesh = fire.Mesh(
            meshfname,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    
    return mesh

def meshing(frequency, model_data, comm, out_name = 'fwi', input_model_name=None):
    # try 100:
    vp0_filename, vp0_filetype = get_file_type(input_model_name)
    if vp0_filetype == ".hdf5" or vp0_filetype == ".segy":
        input_model_name = vp0_filename

    f_name = str(int(frequency))
    if input_model_name == None:
        input_model_name = "velocity_models/"+f_name+"Hz"

    input_vp_model = input_model_name+".segy"
    output_mesh_name = "meshes/"+out_name+"_"+f_name+"Hz"

    model = {}
    model['inversion'] ={
        'initial_guess': input_model_name+".hdf5",
    }
    model["mesh"] = {
        "Lz": model_data["mesh"]["Lz"],  # depth in km - always positive
        "Lx": model_data["mesh"]["Lx"],  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": output_mesh_name,
        "initmodel": input_model_name+".hdf5",
        "truemodel": "not_used.hdf5",
    }
    model["BCs"] = model_data["BCs"]

    vp_filename, vp_filetype = os.path.splitext(input_vp_model)
    
    write_velocity_model(input_vp_model, ofname = vp_filename)
    new_vpfile = vp_filename+'.hdf5'

    s_print('Entering mesh generation', comm)

    mesh_filename = output_mesh_name

    mesh = build_mesh(model, frequency, mesh_filename, vp_filename+'.segy', comm )

    show = True
    if show == True:
        V = fire.FunctionSpace(mesh, 'KMV', 4)
        vp = spyro.io.interpolate(model, mesh, V, guess=True)
        outputfile = fire.File('real_vp.pvd')
        outputfile.write(vp)
    # comm.barrier()
    return True

def meshes_for_real_shots(frequencies, model_data, comm):
    input_model_name = 'velocity_models/cut_marmousi'
    for frequency in frequencies:
        meshing(frequency, model_data, comm, input_model_name=input_model_name, out_name='real')

    return True

def s_print(str, comm):
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(str, flush = True)


frequencies = [3.0, 5.0, 8.0]
number_of_iterations=2

real_model = "velocity_models/cut_marmousi.hdf5"
initial_guess = "velocity_models/cut_marmousi_400.hdf5"
building_real_meshes = False

model = create_model()
comm = spyro.utils.mpi_init(model)

model['inversion']['true_model'] = real_model
if building_real_meshes == True:
    meshes_for_real_shots(frequencies, model, comm)

comm.comm.barrier()

cont = 0
for frequency in frequencies:
    model["acquisition"]["frequency"] = frequency
    s_print(f"Starting with frequency {frequency} on frequency band {cont+1}.",comm)
    ## Getting initial guess for this frequency band and generating guess mesh
    if cont ==0:
        initial_guess_current = initial_guess
        model["inversion"]["initial_guess"] = initial_guess_current
        meshing(frequency, model, comm , input_model_name= initial_guess_current )

    else:
        meshing(frequency, model, comm , input_model_name= initial_guess_current )
        
    ## Running forward problem
    s_print("Generating shot record.",comm)
    model["mesh"]['meshfile'] = 'meshes/real_'+str(int(frequency))+'Hz.msh'
    comm.comm.barrier()
    comm.ensemble_comm.barrier()
    p_r = forward_solver(model, comm, save_shots = True,guess=False)

    ## Runing FWI
    s_print(f"Starting FWI with {number_of_iterations} iterations.",comm)
    model["mesh"]['meshfile'] = 'meshes/fwi_'+str(int(frequency))+'Hz.msh'

    comm.comm.barrier()
    comm.ensemble_comm.barrier()
    segy_fname=fwi_solver(model, comm, number_of_iterations=number_of_iterations)
    initial_guess_current = segy_fname
    cont+=1


