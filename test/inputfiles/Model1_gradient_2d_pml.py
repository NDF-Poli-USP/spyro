import os
import spyro

fname = os.path.join(os.path.dirname(__file__), "../meshes/Uniform2D_pml")

model_pml = {}

# Define mesh file to be used:
meshfile = fname

# Define initial velocity model:
truemodel = "not_used"
initmodel = "not_used"

opts = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}
parallelism = {
    "type": "automatic",
}
mesh = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": meshfile + ".msh",
    "initmodel": initmodel + ".hdf5",
    "truemodel": truemodel + ".hdf5",
}
BCs = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 1.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.20,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.20,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

acquisition = {
    "source_type": "Ricker",
    "source_pos": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((-0.95, 0.1), (-0.95, 0.9), 100),
}
timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 99999,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}

aut_dif = {
    "status": False, 
}

model_pml = {
    "self": None,
    "opts": opts,
    "BCs": BCs,
    "parallelism": parallelism,
    "mesh": mesh,
    "acquisition": acquisition,
    "timeaxis": timeaxis,
    "aut_dif": aut_dif,
}
