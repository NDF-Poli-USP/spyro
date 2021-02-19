import os
import spyro

fname = os.path.join(os.path.dirname(__file__), "../meshes/Uniform3D")

# Define mesh file to be used:
meshfile = fname

# Define initial velocity model:
truemodel = "not_used"
initmodel = "not_used"


# Choose method and parameters
opts = {
    "method": "KMV",
    "variant": None,
    "quadrature": "KMV",
    "degree": 3,  # p order
    "dimension": 3,  # dimension
}

parallelism = {
    "type": "automatic",  # options: automatic, custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

mesh = {
    "Lz": 0.75,  # depth in km - always positive
    "Lx": 0.50,  # width in km - always positive
    "Ly": 0.50,  # thickness in km - always positive
    "meshfile": meshfile + ".msh",
    "initmodel": initmodel + ".hdf5",
    "truemodel": truemodel + ".hdf5",
}

PML = {
    "status": True,  # True,  # True or false
    "outer_bc": "none",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 3.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.25,  # thickness of the pml in the y-direction (km) - always positive
}

recvs = spyro.create_2d_grid(0.10, 0.40, 0.10, 0.40, 8)  # 64 receivers
recvs = spyro.insert_fixed_value(recvs, -0.15, 0)  # at 0.15 m deep

acquisition = {
    "source_type": "Ricker",
    "frequency": 10.0,
    "delay": 1.0,
    "num_sources": 1,
    "source_pos": [(-0.10, 0.25, 0.25)],
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.5,  # Final time for event
    "dt": 0.0005,  # timestep size
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
