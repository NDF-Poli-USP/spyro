from firedrake import File
from firedrake import Function
from firedrake import triplot
from movement import *
import matplotlib.pyplot as plt
import spyro
import sys
import time

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    #"type": "automatic",
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "None",
    "initmodel": "None",
    "truemodel": "None",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.9,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}

comm = spyro.utils.mpi_init(model)

# read mesh 0 generated using vp 0
model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp_smoothed_.msh"
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed.segy.hdf5"# m/s
T0, V0 = spyro.io.read_mesh(model, comm)
vp0 = spyro.io.interpolate(model, T0, V0, guess=False)

# read mesh 1 generated using vp 1
model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp.msh"
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
T1, V1 = spyro.io.read_mesh(model, comm) # T1, V1 != T0, V0
vp1 = spyro.io.interpolate(model, T1, V1, guess=False)

# using mesh 0, read vp 1
model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp_smoothed_.msh"
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
T2, V2 = spyro.io.read_mesh(model, comm) # T2, V2 == T0, V0
vp2 = spyro.io.interpolate(model, T2, V2, guess=False)

File("vp0.pvd", comm=comm.comm).write(vp0)
File("vp1.pvd", comm=comm.comm).write(vp1)
File("vp2.pvd", comm=comm.comm).write(vp2)

# build a monitor function on T0, V0
M = Function(V0)
alpha=3
M.dat.data[:] = (vp0.dat.data[:] / vp2.dat.data[:])**alpha # they have the same mesh/space
File("monitor_before_amr.pvd").write(M)

def monitor(mesh):
    return M # FIXME it probably is wrong (this is way there are some shifting of some regions in the Marmousi mesh)

method = "quasi_newton"
tol = 1.0e-03
mover = MongeAmpereMover(T0, monitor, method=method, rtol=tol)
mover.move()

# to print the mesh using Paraview, use as image resolution 3*(520 x 120)

File( "monitor_after_amr_alpha=" + str(alpha) + ".pvd" ).write(mover.phi, mover.sigma)

# prepare data to plot mesh


fig, axes = plt.subplots()
triplot(mover.mesh, axes=axes)
axes.axis(False)
axes.axis("equal")
#plt.tight_layout()
plt.show()
#plt.savefig(f"plots/mesh_{j}.png")

