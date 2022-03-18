from firedrake import *
import spyro
import sys
import time

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 3,  # dimension
}
model["parallelism"] = {
    #"type": "automatic",
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 15.0,  # depth in km - always positive
    "Lx": 35.0,  # width in km - always positive
    "Ly": 40.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 1.6,  # Final time for event (used to reach the bottom of the domain)
    "tf": 0.100,  # Final time for event (used to measure the time)
    #"dt": 0.00025,
    "dt": 0.0001, # needs for P=5
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)

edge_length = 1.
bbox = (-model["mesh"]["Lz"], 0.0, 0.0, model["mesh"]["Lx"], 0.0, model["mesh"]["Ly"])
cube = SeismicMesh.Cube(bbox)
points, cells = SeismicMesh.generate_mesh(
    domain=cube,
    edge_length=edge_length,
    max_iter = 80,
    comm = comm.ensemble_comm,
    verbose = 2
    )


element = spyro.domains.space.FE_method(
                mesh, model["opts"]["method"], model["opts"]["degree"]
            )

V = FunctionSpace(mesh, element)

model["mesh"]["truemodel"] = "velocity_models/seam/Vp.hdf5"# m/s

vp = spyro.io.interpolate(model, mesh, V, guess=False)
File("seam_vp.pvd").write(vp)



