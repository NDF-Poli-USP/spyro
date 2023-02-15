import meshio
import firedrake as fire
import spyro
import time
try:
    import SeismicMesh
except ImportError:
    pass

# Adding model parameters:
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 3,  # p order use 2 or 3 here
    "dimension": 3,  # dimension
}

model["parallelism"] = {
    "type": "automatic",  # options: automatic, spatial, or custom.
    # only if the user wants a different number of cores for every shot.
    "custom_cores_per_shot": [],
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 8.0,  # depth in km - always positive
    "Lx": 8.0,  # width in km - always positive
    "Ly": 8.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": True,  # True or false
    # None or non-reflective (outer boundary condition)
    "outer_bc": "non-reflective",
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 1,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-3,  # theoretical reflection coefficient
    "lz": 0.286,  # thickness of the PML in z-direction (km) - always positive
    "lx": 0.286,  # thickness of the PML in x-direction (km) - always positive
    "ly": 0.286,  # thickness of the PML in y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-4.0, 0.429, 4.0)],
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 216,
    "receiver_locations": spyro.create_3d_grid(
        (-4.715, 2.574, 3.285),
        (-3.285, 4.004, 4.715),
        6),
}

# Simulate for 1.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 1.0,  # Final time for event
    "dt": 0.005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 400,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)

# Starting meshing procedure with seismic mesh. This can be done seperately
# in order to not have to
# when testing multiple cores.

print('Entering mesh generation', flush=True)
if model['opts']['degree'] == 2:
    M = 5.1
elif model['opts']['degree'] == 3:
    M = 3.1

edge_length = 0.286/M
Real_Lz = model["mesh"]["Lz"] + model["BCs"]["lz"]
Lx = model["mesh"]["Lx"]
Ly = model["mesh"]["Ly"]
pad = model["BCs"]["lz"]

bbox = (-Real_Lz, 0.0, -pad, Lx+pad, -pad, Ly+pad)
cube = SeismicMesh.Cube(bbox)
points, cells = SeismicMesh.generate_mesh(
    domain=cube,
    edge_length=edge_length,
    max_iter=80,
    comm=comm.ensemble_comm,
    verbose=2
)

points, cells = SeismicMesh.sliver_removal(
    points=points,
    bbox=bbox,
    max_iter=100,
    domain=cube,
    edge_length=edge_length,
    preserve=True
)

meshio.write_points_cells(
    "meshes/benchmark_3d.msh",
    points,
    [("tetra", cells)],
    file_format="gmsh22",
    binary=False
)

# Mesh generation finishes here.

mesh = fire.Mesh(
    "meshes/benchmark_3d.msh",
    distribution_parameters={
        "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
    },
)

method = model["opts"]["method"]
degree = model["opts"]["degree"]

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print(f"Setting up {method} a {degree}tetra element", flush=True)

element = fire.FiniteElement(
    method,
    mesh.ufl_cell(),
    degree=degree,
    variant="KMV"
)

V = fire.FunctionSpace(mesh, element)

vp = fire.Constant(1.429)

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Finding sources and receivers", flush=True)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Starting simulation", flush=True)

wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

t1 = time.time()
p, p_r = spyro.solvers.forward(
    model,
    mesh,
    comm,
    vp,
    sources,
    wavelet,
    receivers
)
print(time.time() - t1, flush=True)
