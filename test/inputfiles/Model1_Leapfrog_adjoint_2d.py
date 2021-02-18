import os
from spyro import create_transect

fname = os.path.join(os.path.dirname(__file__), "../meshes/Uniform2D")

# Define mesh file to be used:
meshfile = fname

# Define initial velocity model:
truemodel = "not_used"
initmodel = "not_used"


# Choose method and parameters
opts = {
    "method": "KMV",
    "quadrature": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

parallelism = {
    "type": "automatic",  # options: automatic, custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

mesh = {
    "Lz": 3.000,  # depth in km - always positive
    "Lx": 3.000,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": meshfile + ".msh",
    "initmodel": initmodel + ".hdf5",
    "truemodel": truemodel + ".hdf5",
}

PML = {
    "status": False,  # True,  # True or false
    "outer_bc": "None",  # "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 1.000,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 1.000,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

acquisition = {
    "source_type": "Ricker",
    "frequency": 5.0,
    "delay": 1.0,
    "num_sources": 1,
    "source_pos": [(1.5, -0.5)],
    "num_receivers": 501,
    "receiver_locations": create_transect((0.1, -2.90), (2.9, -2.90), 501),
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "nspool": 20,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}  # how freq. to output to files and screen

inversion = {
    "freq_bands": [None]
}  # cutoff frequencies (Hz) for Ricker source and to low-pass the observed shot record


# Create your model with all the options
model = {
    "self": None,
    "inversion": inversion,
    "opts": opts,
    "PML": PML,
    "parallelism": parallelism,
    "mesh": mesh,
    "acquisition": acquisition,
    "timeaxis": timeaxis,
}
