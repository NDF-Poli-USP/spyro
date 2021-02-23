import os
import spyro

fname = os.path.join(os.path.dirname(__file__), "../meshes/Uniform2D")

model = {}

opts = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

parallelism = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
}

mesh = {
    "Lz": 0.75,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

PML = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

acquisition = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.1, 0.75)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 1.4), 500),
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.00,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}

model = {
    "self": None,
    "opts": opts,
    "PML": PML,
    "parallelism": parallelism,
    "mesh": mesh,
    "acquisition": acquisition,
    "timeaxis": timeaxis,
}
