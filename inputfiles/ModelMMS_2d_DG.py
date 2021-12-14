import os

fname = os.path.join(os.path.dirname(__file__), "Model1")

# Define mesh file to be used:
meshfile = fname

# Define initial velocity model:
truemodel = fname
initmodel = fname


# Choose method and parameters
opts = {
    "method": "DG",
    "variant": None,
    "type": "SIP",  # for DG only - SIP, NIP and IIP
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "mesh_size": 0.005,  # h
    "beta": 0.0,  # for Newmark only
    "gamma": 0.5,  # for Newmark only
}

parallelism = {
    "type": "automatic", # options: automatic, custom, off
    "custom_cores_per_shot": [], # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots. 
    }

mesh = {
    "Lz": 2.000,  # depth in km - always positive
    "Lx": 3.00000,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": meshfile + ".msh",
    "initmodel": initmodel + ".hdf5",
    "truemodel": truemodel + ".hdf5",
}

PML = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.250,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.250,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

acquisition = {
    "source_type": "MMS",
    "num_sources": 1,
    "frequency": 2.0,
    "delay": 1.0,
    "source_pos": [(-0.05, 1.5)],
    "num_receivers": 256,
    "receiver_locations": [()],
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.4,  # Final time for event
    "dt": 0.001,  # timestep size
    "nspool": 20,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
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
