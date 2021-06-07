from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
)

from firedrake import *
from firedrake_adjoint import *
import spyro
import copy
import numpy as np
from spyro.domains import quadrature
import math
import numpy                  as np
import matplotlib.pyplot      as plot
import matplotlib.ticker      as mticker  
from matplotlib               import cm, ticker
from mpl_toolkits.axes_grid1  import make_axes_locatable

model = {}
model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    "num_cores_per_shot": 1
}
model["mesh"] = {
    "Lz": 1.,  # depth in km - always positive
    "Lx": 1.,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/square.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.0,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": spyro.create_transect((0.1, 0.1), (0.1, 0.9), 1),
    "frequency": 3.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 0.9), 100),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.001,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
num_rec = model["acquisition"]["num_receivers"]
δs = np.linspace(0.1, 0.9, num_rec)
X, Y = np.meshgrid(-0.1, δs)
xs = np.vstack((X.flatten(), Y.flatten())).T

comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
def _make_vp_exact(V, mesh):
    x, y = SpatialCoordinate(mesh)
    velocity = conditional(x > -0.5, 1.5, 3)
    vp = Function(V, name="velocity").interpolate(velocity)
    File("velocity.pvd").write(vp)
    return vp

vp_exact = _make_vp_exact(V,mesh)

sources = spyro.Sources(model, mesh, V, comm)

solver  = spyro.solver_AD()

p_rec_exact  = solver.forward_AD(model, mesh, comm,
                            vp_exact, sources, xs) 

spyro.io.save_shots(model, comm, p_rec_exact,file_name='true_rec_' + str(0))
