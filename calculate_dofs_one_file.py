from firedrake import *
import numpy as np
import spyro

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


comm = spyro.utils.mpi_init(generate_model(3))

if comm.comm.rank == 0:
    print(f"For hom", flush = True)
mesh = Mesh(
    "meshes/hom_overthrust.msh",
    comm=comm.comm,
    distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.NONE, 0)
        },
    )
element = FiniteElement('KMV', mesh.ufl_cell(), degree=3, variant = 'KMV')

V = FunctionSpace(mesh, element)
comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("DoFs as Vdim", flush=True)
print(V.dim(), flush = True)
comm.comm.barrier()
u = Function(V)
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("DoFs as data with halos", flush=True)
comm.comm.barrier()
udat = u.dat.data_with_halos[:]
print(np.shape(udat), flush = True)
comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("DoFs as data", flush=True)
comm.comm.barrier()
udat = u.dat.data[:]
print(np.shape(udat), flush = True)
comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("\n\n\n", flush = True)
