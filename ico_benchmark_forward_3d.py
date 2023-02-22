from firedrake import File
import firedrake as fire
import time
import numpy as np
import spyro
import sys

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

def create_model(scale):
    degree = 3
    method = "KMV"
    quadrature = "KMV"

    lbda = 1.429/5.0
    pad = lbda
    Lz = 8.
    Real_Lz = Lz+ pad
    Lx = 8.
    Real_Lx = Lx+ 2*pad
    Ly = 8.
    Real_Ly = Ly + 2*pad

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
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV",  # Equi or KMV
        "degree": 3,  # p order use 2 or 3 here
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
    model["aut_dif"]={
	"status": False
    }

    return model



scale = int(sys.argv[1])

receivers = spyro.create_2d_grid(0.25, 7.25, 0.25, 7.25, 30)
receivers = spyro.insert_fixed_value(receivers, -0.15, 0)

# Adding model parameters:
model = create_model(scale)


comm = spyro.utils.mpi_init(model)

comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Reading mesh", flush=True)

mesh = fire.Mesh(
            "meshes/ico_scaling_mesh.msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
comm.comm.barrier()


t1 = time.time()
element = fire.FiniteElement(model["opts"]["method"], mesh.ufl_cell(), degree=model["opts"]["degree"], variant="KMV")
V = fire.FunctionSpace(mesh, element)
vp = fire.Constant(1.429)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Starting simulation", flush=True)

t2 = time.time()
p, p_r = spyro.solvers.forward(
    model, mesh, comm, vp, sources, wavelet, receivers, output=False
)
t3 = time.time()

print(f"Time without forward problem: {t2-t1}", flush = True)
print(f"Time with only forward problem: {t3-t2}", flush = True)
print(f"Total time problem: {t3-t1}", flush = True)

comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print(f"END of forward problem", flush = True)

