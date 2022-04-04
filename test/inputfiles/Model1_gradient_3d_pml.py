import os
import spyro

fname = os.path.join(os.path.dirname(__file__), "../meshes/Uniform3D_pml")

model_pml = {}

# Define mesh file to be used:
meshfile = fname

# Define initial velocity model:
truemodel = "not_used"
initmodel = "not_used"

opts = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 3,  # dimension
}
parallelism = {
    "type": "automatic",
}
mesh = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 1.0,  # thickness in km - always positive
    "meshfile": meshfile + ".msh",
    "initmodel": initmodel + ".hdf5",
    "truemodel": truemodel + ".hdf5",
}
BCs = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.20,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.20,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.20,  # thickness of the PML in the y-direction (km) - always positive
}

receivers = spyro.insert_fixed_value(spyro.create_2d_grid(0.1,0.9,0.1,0.9,10),-0.9, 0)
print(len(receivers))

acquisition = {
    "source_type": "Ricker",
    "source_pos": [(-0.1, 0.5, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": receivers, 
}
timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.0001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}

model_pml = {
    "self": None,
    "opts": opts,
    "BCs": BCs,
    "parallelism": parallelism,
    "mesh": mesh,
    "acquisition": acquisition,
    "timeaxis": timeaxis,
}
