from firedrake import *

import time
import json
import argparse
import spyro
import spyro.optimizers.damp as adam

from mpi4py import MPI
import numpy.linalg as la
import numpy as np
import cplex

# Load parameters
model = spyro.io.load_model()

# Create the computational environment
comm = spyro.utils.mpi_init(model)

# Create mesh
mesh, V = spyro.utils.create_mesh(model, comm, quad=False)
print(f"Function space with {V.dim()} DoFs")

# load data
if model["data"].get("initfile"):
    vp_guess = spyro.utils.load_velocity_model(
        model, V, source_file=model["data"]["initfile"]
    )
else:
    vp_guess = interpolate(Constant(model["opts"]["cmin"]), V)
File("vp_init.pvd").write(vp_guess)
vp_gradient = Function(V)

# get control from vp
normalized_vp = spyro.utils.normalize_vp(model, vp_guess)
discrete_data = normalized_vp.dat.data.round()
normalized_vp.dat.data[:] = discrete_data
File("vp_normalized.pvd").write(normalized_vp)

# acquisition geometry
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

# excitation frequency
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
                    # model, mesh, comm, vp_guess, sources, receivers,
                    # source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model,
                        normalized_vp), sources, receivers, source_num=sn)
                # Calculate the misfit.
                misfit = spyro.utils.evaluate_misfit( model, comm,
                        p_guess_recv, p_exact_recv)
                # Calculate the gradient of the functional.
                dJ = spyro.solvers.Leapfrog_adjoint(
                    # model, mesh, comm, vp_guess, p_guess, misfit,
                    # source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model,
                        normalized_vp), p_guess, misfit, source_num=sn)
                dJ_local += dJ
                # Calculate the L2-functional.
                J = spyro.utils.compute_functional(model, comm, misfit)
                J_local[0] += J

        # if comm.ensemble_comm.rank == 0:
        #     # Plot updated control variable
        #     viz_m_file.write(normalized_vp)
        #     # Plot updated P-wave velocity
        #     viz_vp_file.write(vp_guess)

    # Sum functional and gradient over ensemble members
    comm.ensemble_comm.Allreduce(dJ_local, dJ_total, op=MPI.SUM)
    comm.ensemble_comm.Allreduce(J_local, J_total, op=MPI.SUM)

    # Mask the vertices that are in the water
    # dJ_total[water]=0.0
    dJ_total_io = Function(V, val=dJ_total)

    # Filter gradient
    if model["cplex"]["use_rmin"]: dJ_total_io = spyro.utils.helmholtz_filter(
            dJ_total_io, model['opts']['rmin'])
    dJ_total = dJ_total_io.dat.data[:]
    dJ_total[water]=0.0

    vp_gradient.assign(dJ_total_io)
    # if comm.ensemble_comm.rank == 0:
    #     viz_dm_file.write(vp_gradient)

    # write paraview output
    cb.write_file(m=normalized_vp, dm=vp_gradient, vp=vp_guess)

    return J_total[0], dJ_total


# if comm.ensemble_comm.rank == 0:
#     # Initialize a file to visualize the control/vp each iteration
#     viz_dm_file = File("gradient_cplex" + str(FREQ) + ".pvd", comm=comm.comm)
#     viz_m_file = File("control_" + str(FREQ) + ".pvd", comm=comm.comm)
#     viz_m_file.write(normalized_vp)
#     viz_vp_file = File("vp_guess" + str(FREQ) + ".pvd", comm=comm.comm)
#     viz_vp_file.write(vp_guess)

# Callback object for output files
cb = spyro.io.Callback(model, comm)
cb.create_file(normalized_vp, vp_gradient, vp_guess)
# Gather the full velocity data on the master rank
xi = normalized_vp.vector().gather()
# Bounds for control
if model["material"]["type"] == "simp":
    lb = 0
    ub = 1
elif model["material"]["type"] == None:
    lb = model["opts"]["cmin"]
    ub = model["opts"]["cmax"]

stop = [0]
change = 100
counter = 0
beta = model["cplex"]["beta"]
mul_beta = model["cplex"]["mul_beta"]
mul_rmin = model["cplex"]["mul_rmin"]
lim_rmin = model["cplex"]["lim_rmin"]
iter_info = "it.: {:d} | obj.f.: {:e} | rel.var.: {:2.2f}% |"
iter_info += " move: {:g} | rmin: {:g}"

# ADAM parameters
gamma_m = model["cplex"].get("gamma_m", 0.5)
gamma_v = model["cplex"].get("gamma_v", 0.5)
m, v = 0, 0

# Call the optimization routine from the master rank.
if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
   
    while counter < 200:

        # evaluate functional, gradient
        model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'], root=0)
        J, dJ = shots(xi, stop)
        # dJ /= la.norm(dJ)

        # get first moment
        m = adam.moving_avg(gamma_m, m, dJ)
        # second moment
        v = adam.moving_avg(gamma_v, v, dJ ** 2)

        # correct for bias
        m_ = adam.remove_bias(gamma_m, m, counter)
        v_ = adam.remove_bias(gamma_v, v, counter)

        # evaluate RMSProp averaged gradient
        dJ = adam.RMSprop(m_, v_)

        # exit if close enough, or if objective function stagnates
        if J < 1e-16 or (counter > 0 and J == J0): break

        if counter > 0:
            # calcula a variação na função objetivo
            change = (J - J0) / J0

                # update beta
        beta = spyro.optimizers.update_flip_limits( beta, counter, mul_beta,
                change, xi, mode='counter')
        # update rmin
        model['opts']['rmin'] = spyro.optimizers.update_rmin(
                model['opts']['rmin'], counter, lim_rmin, mul_rmin)

        # update control
        # xi[:] = spyro.optimizers.iterate_cplex(dJ, beta, xi)
        dxi = spyro.optimizers.iterate_cplex_linear(dJ, beta, xi)
        xi += dxi

        # printa diferença entre iterações
        print("\n"+iter_info.format( counter, J, change, beta,
            model['opts']['rmin'])+"\n")

        # save old functional
        J0, dJ0 = J, dJ

        counter += 1

    stop = [1]
    model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'],root=0)
    shots(xi, stop)

    print("\n"+iter_info.format( counter, J, change, beta,
            model['opts']['rmin'])+"\n")

else:
    while stop[0] == 0:
        model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'], root=0)
        shots(xi, stop)

# Retrieve values
spyro.utils.spatial_scatter(comm, xi, normalized_vp)
normalized_vp.dat.data[:] = xi[:]
vp_guess = spyro.utils.control_to_vp(model, normalized_vp)

# Save'em
spyro.utils.save_velocity_model(comm, vp_guess, model["data"]["resultfile"])
