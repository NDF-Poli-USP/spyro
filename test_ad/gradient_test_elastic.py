from firedrake import *
from scipy.optimize import *
from firedrake_adjoint import *
from pyadjoint import enlisting
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

#from ..domains import quadrature, space

#parameters from Daiane
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "MMV", # Equi or KMV #FIXME it will be removed
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
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((-0.1, 0.6), (-0.1, 1.4), 1),
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 10, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((-0.2, 0.6), (-0.2, 1.4), 10),
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
    raise ValueError("VertexOnlyMesh does no worh with meshes which coordinates were modified") 
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
    with stop_annotating():
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

            lamb = Function(H).interpolate(_lamb/2 + 0.0 * x)
            mu = Function(H).interpolate(_mu/2 + 0.0 * x)
            # lamb  = Function(H).interpolate(
            #     2.25
            #     + 2. * tanh(20 * (0.125 - sqrt(( x + 1) ** 2 + (z - 0.5) ** 2)))
            # )
            # mu  = Function(H).interpolate(
            #     2.5
            #     + 1.5 * tanh(20 * (0.125 - sqrt(( x + 1) ** 2 + (z - 0.5) ** 2)))
            # )
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
point_cloud = receivers.set_point_cloud(comm)
#sys.exit("exit")

print("Run exact model", flush=True)
u_exact, uz_exact, ux_exact, uy_exact = spyro.solvers.forward_elastic_waves_AD(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, point_cloud, output=False
)
u_exact_rec = [uz_exact, ux_exact]

print("Run guess model", flush=True)
u_guess, uz_guess, ux_guess, uy_guess, J = spyro.solvers.forward_elastic_waves_AD(
    model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, point_cloud, output=False,
    true_rec=u_exact_rec, fwi=True
)

File("u_exact.pvd").write(u_exact[-1])
File("u_guess.pvd").write(u_guess[-1])
#sys.exit("exit")

print("Compute the gradient using AD",flush=True)
control_lamb  = Control(lamb_guess)
control_mu    = Control(mu_guess)


dJdl_local = compute_gradient(J, control_lamb, options={"riesz_representation": "L2"})
dJdm_local = compute_gradient(J, control_mu, options={"riesz_representation": "L2"})

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
Jhat_l     = ReducedFunctional(J, control_lamb) 
Jhat_m     = ReducedFunctional(J, control_mu) 

with stop_annotating():

    steps = [1e-4, 1e-5, 1e-6]  # step length

    delta_l = Function(H)  # model direction (random)
    delta_l.assign(dJdl)
    delta_m = Function(H)  # model direction (random)
    delta_m.assign(dJdm)
    # delta_m = Function(V)  # model direction (random)
    # delta_m.assign(dJdm)
    derivative_l = enlisting.Enlist(Jhat_l.derivative())
    derivative_m = enlisting.Enlist(Jhat_m.derivative())
    # derivative_m = enlisting.Enlist(Jhat_m.derivative())
    hs_l = enlisting.Enlist(delta_l)
    hs_m = enlisting.Enlist(delta_m)
    # hs_m = enlisting.Enlist(delta_m)

    projnorm_lamb = sum(hi._ad_dot(di) for hi, di in zip(hs_l, derivative_l))
    projnorm_mu   = sum(hi._ad_dot(di) for hi, di in zip(hs_m, derivative_m))
    # projnorm_mu   = sum(hi._ad_dot(di) for hi, di in zip(hs_m, derivative_m))
    # this deepcopy is important otherwise pertubations accumulate
    lamb_original = lamb_guess.copy(deepcopy=True)
    mu_original   = mu_guess.copy(deepcopy=True)
    # mu_original   = mu_guess.copy(deepcopy=True)

    print('######## Computing the gradient by finite diferences ########')
    errors = []
    for step in steps:  # range(3):
        lamb_guess = lamb_original + step * delta_l
        mu_guess = mu_original #+ step * delta_m FIXME
        # mu_guess   = mu_original #+ step * delta_m FIXME

        u_guess, uz_guess, ux_guess, uy_guess, Jp = spyro.solvers.forward_elastic_waves_AD(
            model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, point_cloud, output=False,
            true_rec=u_exact_rec, fwi=True
            )


        fd_grad = (Jp - J) / step
        print(
            "\n Cost functional for step "
            + str(fd_grad)
            + ", grad'*dir (lambda) : "
            + str(projnorm_lamb)
            + ", grad'*dir (mu) : "
            + str(projnorm_mu)
            # + ", grad'*dir (mu) : "
            # + str(projnorm_mu)
            + " \n ",
        )

