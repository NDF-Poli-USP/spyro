import meshio
import firedrake as fire
import spyro
import time
import sys

## Adding model parameters:
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order use 2 or 3 here
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic, spatial, or custom.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 114.4,  # depth in km - always positive
    "Lx": 85.8,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "none",  # polynomial, hyperbolic, shifted_hyperbolic
    "lz": 0.286,  # thickness of the absorbing layer in the z-direction (km) - always positive
    "lx": 0.286,  # thickness of the absorbing layer in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the absorbing layer in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-5.72, 4.29)],
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 15,
    "receiver_locations": spyro.create_transect(
        (-5.72,  6.29),
        (-5.72, 14.29),
        15),
}

# Simulate for 1.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 4.0,  # Final time for event
    "tf": 0.05,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 9999999,  # how frequently to output solution to pvds
    "fspool": 9999999,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)

if 0: # mesh generation with SeismicMesh {{{
    import SeismicMesh
    ## Starting meshing procedure with seismic mesh. This can be done seperately in order to not have to
    # when testing multiple cores.

    print('Entering mesh generation', flush = True)
    if model['opts']['degree']   == 2:
        M = 5.85
    elif model['opts']['degree'] == 3:
        M = 3.08
    elif model['opts']['degree'] == 4:
        M = 2.22
    elif model['opts']['degree'] == 5:
        M = 1.69

    edge_length = 0.286/M
    Real_Lz = model["mesh"]["Lz"] + model["BCs"]["lz"]
    Lx = model["mesh"]["Lx"]
    pad = model["BCs"]["lz"]

    bbox = (-Real_Lz, 0.0,-pad, Lx+pad)
    rec = SeismicMesh.Rectangle(bbox)
    if comm.comm.rank == 0:
        points, cells = SeismicMesh.generate_mesh(
            domain=rec, 
            edge_length=edge_length, 
            comm = comm.ensemble_comm,
            verbose = 0
            )

        points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
            
        meshio.write_points_cells("meshes/benchmark_2d_elastic_waves_P" + str(model['opts']['degree']) + ".msh",
            points,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )

    sys.exit("exit without running")
    ## Mesh generation finishes here.
#}}}

mesh = fire.Mesh(
    "meshes/benchmark_2d_elastic_waves_P" + str(model['opts']['degree']) + ".msh",
    distribution_parameters={
        "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
    },
)

method = model["opts"]["method"]
degree = model["opts"]["degree"]

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print(f"Setting up {method} a {degree}tetra element", flush=True)

element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")

V = fire.FunctionSpace(mesh, element)

vs = fire.Constant(1.429) # mesh for elastic waves is built on vs
vp = fire.Constant(1.5*1.429)
rho = fire.Constant(1.0)
mu = rho * vs ** 2.
lamb = rho * vp ** 2. - 2 * mu

if 0: # print mu and lamb {{{
    V2 = fire.VectorFunctionSpace(mesh, element)
    print("DOF: "+str(V2.dof_count),flush=True)
    print("Cells: "+str(mesh.num_cells()),flush=True)
    #fire.File("lamb.pvd").write( fire.Function(V, name="lamb").interpolate(lamb) )
    #fire.File("mu.pvd").write( fire.Function(V, name="mu").interpolate(mu) )
    sys.exit("exit without running")
#}}}

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
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
print(time.time() - t1, flush=True)
