import firedrake as fire
import spyro
from spyro.full_waveform_inversion.fwi import simpleFWI

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",
}
model["inversion"] = {
    "regularization" : True,
    "gamma" : 1e-4,
    "gradient_regularization" : False,
    "shot_record" : False,
}
model["mesh"] = {
    "Lz": 1.0,   # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,   # thickness in km - always positive
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.1,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.1,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.5, 0.8)],#[(-0.4,0.8),(-0.5,0.8),(-0.6,0.8)],
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 10,
    "receiver_locations": spyro.create_transect((-0.2, 0.2), (-0.8, 0.2),10),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.50,  # Final time for event
    "dt": 0.0005,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 20,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}


mesh = fire.UnitSquareMesh(30,30,diagonal='crossed')
mesh.coordinates.dat.data[:, 0] *= -1.0
x, y = fire.SpatialCoordinate(mesh)
element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
V = fire.FunctionSpace(mesh, element)

vp_initial = fire.Function(V).interpolate(fire.Constant(2.0))

vp_true = fire.Function(V).interpolate(fire.conditional(y<0.5, 1.0, 3.0))
iteration_limit = 50
vp = simpleFWI(model, iteration_limit, mesh, vp_initial, vp_true, generate_shot= True)

