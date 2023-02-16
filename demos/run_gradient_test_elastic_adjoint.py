from firedrake import *
from scipy.optimize import *
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
#from ..domains import quadrature, space

#parameters from Daiane
model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV #FIXME it will be removed
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    #"type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    "type": "spatial",  #
    #"custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    #"num_cores_per_shot": 1 #FIXME this is not used
}

model["mesh"] = {
    "Lz": 2.0,  # depth in km - always positive
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/square.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": True,  # True or False, used to turn on any type of BC
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": 7.0,
    "delay": 1.0, # FIXME check this
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.6, -0.1), (1.4, -0.1), 4),
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 10, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.6, -0.2), (1.4, -0.2), 10),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0, # Final time for event
    "dt": 0.001,  # timestep size
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

# build or read a mesh {{{
if 0:
    mesh = RectangleMesh(45, 45, model["mesh"]["Lx"], model["mesh"]["Lz"]-0.5, diagonal="crossed", comm=comm.comm)
    mesh.coordinates.dat.data[:, 0] -= 0.0 # PML size
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]-0.5
    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )
    V = VectorFunctionSpace(mesh, element)
    P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
    H = FunctionSpace(mesh, element)
else:
    mesh, H = spyro.io.read_mesh(model, comm) #FIXME update io read mesh for elastic
    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )
    V = VectorFunctionSpace(mesh, element)
    P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
#}}}
# make lamb and mu {{{
def _make_elastic_parameters(H, mesh, guess=False):
    """creating velocity models"""
    x,z = SpatialCoordinate(mesh)
    _cp = 1.5
    _cs = 1.
    _rho = 1.
    _mu = (_cs**2)*_rho # for cp=1.5, cs=1.0
    _lamb = (_cp**2)*_rho-2*_mu
    _cp = 3.5 
    _cs = 2.
    _mu_max = (_cs**2)*_rho # for cp=3.5, cs=2.0
    _lamb_max = (_cp**2)*_rho-2*_mu_max
    if guess:
        lamb = Function(H).interpolate(_lamb + 0.0 * x)
        mu   = Function(H).interpolate(_mu + 0.0 * x)
        File("guess_lamb.pvd").write(lamb)
        File("guess_mu.pvd").write(mu)
    else:
        lamb  = Function(H).interpolate(
            2.25
            + 2. * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
        )
        mu  = Function(H).interpolate(
            2.5
            + 1.5 * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
        )
        File("exact_lamb.pvd").write(lamb)
        File("exact_mu.pvd").write(mu)

    rho = Constant(_rho)
    return lamb, mu, rho
#}}}

lamb_guess, mu_guess, rho = _make_elastic_parameters(H, mesh, guess=True)
lamb_exact, mu_exact, _ = _make_elastic_parameters(H, mesh, guess=False)

sources = spyro.Sources(model, mesh, H, comm)
receivers = spyro.Receivers(model, mesh, H, comm)
wavelet = spyro.full_ricker_wavelet(
            dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
          )

use_AD_type_interp = True
if use_AD_type_interp:
        receivers = receivers.setPointCloudRec(comm, paralel_z=True)
print("Run exact model",flush=True)
u_exact, uz_exact, ux_exact, uy_exact = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=False,
    use_AD_type_interp=use_AD_type_interp # to compare against AD
)
print("Run guess model",flush=True)
u_guess, uz_guess, ux_guess, uy_guess = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, receivers, output=False,
    use_AD_type_interp=use_AD_type_interp # to compare against AD
)
#sys.exit("exit")
File("u_exact.pvd").write(u_exact[-1])
File("u_guess.pvd").write(u_guess[-1])

if use_AD_type_interp:
    uz_guess = np.array(uz_guess)
    ux_guess = np.array(ux_guess)
    uz_exact = np.array(uz_exact)
    ux_exact = np.array(ux_exact)

print("Compute misfit",flush=True)
#J_scale = sqrt(1.e14)
J_scale = sqrt(1.)
misfit_uz = J_scale * spyro.utils.evaluate_misfit(model, uz_guess, uz_exact)
misfit_ux = J_scale * spyro.utils.evaluate_misfit(model, ux_guess, ux_exact)
J_total = np.zeros((1))
J_total[0] += spyro.utils.compute_functional(model, misfit_uz)
J_total[0] += spyro.utils.compute_functional(model, misfit_ux)
J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
if comm.comm.size > 1:
    J_total[0] /= comm.comm.size # paralelismo espacial

print("Run adjoint model",flush=True)

if use_AD_type_interp:
    receivers = spyro.Receivers(model, mesh, H, comm)

misfit_uy = []
dJdl_local, dJdm_local = spyro.solvers.gradient_elastic_waves(
    model, mesh, comm, rho, lamb_guess, mu_guess,
    receivers, u_guess, misfit_uz, misfit_ux, misfit_uy, output=True,
)
dJdl = Function(H, name="dJdl")
dJdm = Function(H, name="dJdm")
if comm.ensemble_comm.size > 1:
    comm.allreduce(dJdl_local, dJdl)
    comm.allreduce(dJdm_local, dJdm)
else:
    dJdl = dJdl_local
    dJdm = dJdm_local
dJdl /= comm.ensemble_comm.size
dJdm /= comm.ensemble_comm.size
if comm.comm.size > 1:
    dJdl /= comm.comm.size
    dJdm /= comm.comm.size

#File("dJdl_elastic_adjoint_gaussian_func.pvd").write(dJdl)
#File("dJdm_elastic_adjoint_gaussian_func.pvd").write(dJdm)
File("dJdl_elastic_adjoint_point_source.pvd").write(dJdl)
File("dJdm_elastic_adjoint_point_source.pvd").write(dJdm)

