"""This python script runs FWI for a default problem.
A different setup may be loaded from a configuration file.
"""
# Import built-in modules
import argparse
import json

from firedrake import *

import spyro

from scipy import optimize
from mpi4py import MPI
import numpy as np

# Load parameters
model = spyro.io.load_model()

# Create the computational environment
comm = spyro.utils.mpi_init(model)

# Create mesh
quad = model["mesh"]["quad"]
mesh, V = spyro.utils.create_mesh(model, comm, quad=quad)
print(f"Function space with {V.dim()} DoFs")

# vp_guess = Function(V)
# vp_guess = interpolate(Constant((model["opts"]["cmin"] + model["opts"]["cmax"]) / 2), V)
vp_guess = interpolate(Constant(model["opts"]["cmin"]), V)
vp_gradient = interpolate(Constant(0), V)

File("vp_init.pvd").write(vp_guess)

normalized_vp = spyro.utils.normalize_vp(model, vp_guess)

File("vp_init_normalized.pvd").write(normalized_vp)

sources, receivers = spyro.Geometry(model, mesh, V, comm).create()

# get water dofs
water = spyro.utils.water_layer(
    mesh, 
    V,  
    vp_guess,
    model
)

# Set water layer to zero
normalized_vp.dat.data[water] = 0

FREQ = model["acquisition"]["frequency"]

# Define a callback function that returns the gradient and functional
def shots(xi, stops):
    """A callback function that returns gradient of control
    and functional to be optimized using scipy

    Parameters
    ----------
    xi: array-like
        The control vector to be optimized
    stops: list of integer
        0 to terminate optimization, 1 to continue

    Returns
    -------
    J: float
        Functional
    dJ: array-like
        The gradient of the functional w.r.t. to the velocity model

    """
    # Spatial communicator rank and size.
    rank = comm.comm.rank
    size = comm.comm.size

    # Update control xi from rank 0.
    xi = COMM_WORLD.bcast(xi, root=0)

    # Update the local vp_guess/control function
    # NOTE TO SELF: this should become a func
    # n = len(vp_guess.dat.data[:])
    n = len(normalized_vp.dat.data[:])
    N = [comm.comm.bcast(n, r) for r in range(size)]
    indices = np.insert(np.cumsum(N), 0, 0)
    # vp_guess.dat.data[:] = xi[indices[rank] : indices[rank + 1]]
    normalized_vp.dat.data[:] = xi[indices[rank] : indices[rank + 1]]

    # Check if the program has converged (and exit if so).
    stops[0] = COMM_WORLD.bcast(stop[0], root=0)

    dJ_total = np.zeros((len(xi),), dtype=float)
    dJ_local = np.zeros((len(xi),), dtype=float)
    J_local = np.array([0.0])
    J_total = np.array([0.0])

    if stops[0] == 0:
        for sn in range(model["acquisition"]["num_sources"]):
            # if sn != 1: continue

            # if sn != 0:
            #     print(f"skipping {sn}")
            #     continue

            if spyro.io.is_owner(comm, sn):
                # Load in "exact" or "observed" shot records from a pickle.
                shotfile = model["data"]["shots"]+str(FREQ)+"Hz_sn_"+str(sn)+".dat"
                p_exact_recv = spyro.io.load_shots(shotfile)

                # Compute the forward simulation for "guess".
                p_guess, p_guess_recv = spyro.solvers.Leapfrog(
                    # model, mesh, comm, vp_guess, sources, receivers, source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model, normalized_vp), sources, receivers, source_num=sn
                )
                # Calculate the misfit.
                misfit = spyro.utils.evaluate_misfit(
                    model, comm, p_guess_recv, p_exact_recv
                )
                # Calculate the gradient of the functional.
                dJ = spyro.solvers.Leapfrog_adjoint(
                    # model, mesh, comm, vp_guess, p_guess, misfit, source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model, normalized_vp), p_guess, misfit, source_num=sn
                )
                dJ_local += dJ
                # Calculate the L2-functional.
                J = spyro.utils.compute_functional(model, comm, misfit)
                J_local[0] += J

    # Sum functional and gradient over ensemble members
    comm.ensemble_comm.Allreduce(dJ_local, dJ_total, op=MPI.SUM)
    comm.ensemble_comm.Allreduce(J_local, J_total, op=MPI.SUM)

    # Mask the vertices that are in the water and assign to Function
    dJ_total[water]=0.0
    vp_gradient.dat.data[:]=dJ_total

    # write paraview output
    cb.write_file(m=normalized_vp, dm=vp_gradient, vp=vp_guess)

    return J_total[0], dJ_total

# Callback object for output files
cb = spyro.io.Callback(model, comm)
cb.create_file(normalized_vp, vp_gradient, vp_guess)
# Gather xi from all ranks
xi = normalized_vp.vector().gather()

# Bounds for control
if model["material"]["type"] == "simp":
    lb = 0
    ub = 1
elif model["material"]["type"] == None:
    lb = model["opts"]["cmin"]
    ub = model["opts"]["cmax"]

# Call the optimization routine from the master rank.
if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
    stop = [0]
    res = optimize.minimize(
        shots,
        xi,
        args=(stop,),  # extra args to call-back function "shots"
        method="L-BFGS-B",
        jac=True,
        bounds=[(lb, ub) for i in range(len(xi))],
        options={"disp": True, "maxiter":200, "gtol": 1e-10},
    )
    stop = [1]
    shots(res.x, stop)
    xi = res.x
else:
    stop = [0]
    while stop[0] == 0:
        shots(xi, stop)

# Retrieve values
xi = COMM_WORLD.bcast(xi, root=0)
spyro.utils.spatial_scatter(comm, xi, normalized_vp)
normalized_vp.dat.data[:] = xi[:]
vp_guess = spyro.utils.control_to_vp(model, normalized_vp)

# Save'em
print("On rank {0}, sum(xi) = {1}".format(
    comm.ensemble_comm.Get_rank(), xi.sum()
    )
)
spyro.utils.save_velocity_model(
    comm, vp_guess, model["data"]["resultfile"])

