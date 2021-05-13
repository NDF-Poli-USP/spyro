import spyro
from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
)
import numpy as np
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
    "truemodel": "velocity_models/MarmousiII_w1KM_EXT_GUESS.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
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
    "source_pos": spyro.create_transect((-0.1, 0.1), (-0.1, 0.9), 2),
    "frequency": 4.0,
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

comm    = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

x, y = SpatialCoordinate(mesh)
velocity = conditional(x > -0.5, 1.5, 3)
vp   = Function(V, name="vp").interpolate(velocity)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
for sn in range(0,model["acquisition"]["num_sources"]):
    solver  = spyro.solver_AD()
    p_r     = solver.forward_AD(model, mesh, comm, vp, sources, wavelet, xs, source_num=sn)
    
    # spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
    spyro.io.save_shots(model, comm, p_r,file_name='true_rec/true_rec_' + str(sn))
