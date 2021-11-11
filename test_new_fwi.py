import spyro
from spyro.full_waveform_inversion.fwi import syntheticFWI

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 5,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",
}
model["inversion"] = {
    "initial_guess" : None,
    "true_model" : "velocity_models/vp_marmousi-ii.segy",
    "cost_functional_regularization" : True,
    "gamma" : 1e-4,
    "gradient_regularization" : False,
}
model["mesh"] = {
    "Lz": 3.5,   # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,   # thickness in km - always positive
    "meshfile": None,
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 2,
    "source_pos": spyro.create_transect((-0.01, 1.0), (-0.01, 15.0), 2),
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 100),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 6.00,  # Final time for event
    "dt": 0.001,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

vp = syntheticFWI(model)

