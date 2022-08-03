from firedrake import File
import firedrake as fire
import spyro

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadrature": "GLL",
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive
    "Lx": 7.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": None,
    "initmodel": None,
    "truemodel": None,
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": None,  #  None or non-reflective (outer boundary condition)
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
    "source_pos": [(-0.1,3.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 7.0), 20),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

