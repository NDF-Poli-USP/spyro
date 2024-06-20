import firedrake as fire
from firedrake import Constant, dx, dot, grad
import numpy as np

from ..io.basicio import parallel_print
from . import helpers
from .. import utils


def central_difference(Wave_object, source_id=0):
    """
    Perform central difference time integration for wave propagation.

    Parameters:
    -----------
    Wave_object: Spyro object
        The Wave object containing the necessary data and parameters.
    source_id: int (optional)
        The ID of the source being propagated. Defaults to 0.

    Returns:
    --------
        tuple:
            A tuple containing the forward solution and the receiver output.
    """
    excitations = Wave_object.sources
    excitations.current_source = source_id
    receivers = Wave_object.receivers
    comm = Wave_object.comm
    temp_filename = Wave_object.forward_output_file

    filename, file_extension = temp_filename.split(".")
    output_filename = filename + "sn" + str(source_id) + "." + file_extension
    if Wave_object.forward_output:
        parallel_print(f"Saving output in: {output_filename}", Wave_object.comm)

    output = fire.File(output_filename, comm=comm.comm)
    comm.comm.barrier()

    X = fire.Function(Wave_object.function_space)

    final_time = Wave_object.final_time
    dt = Wave_object.dt
    t = Wave_object.current_time
    nt = int((final_time - t) / dt) + 1  # number of timesteps

    u_nm1 = Wave_object.u_nm1
    u_n = Wave_object.u_n
    u_np1 = fire.Function(Wave_object.function_space)

    rhs_forcing = fire.Cofunction(Wave_object.function_space.dual())
    usol = [
        fire.Function(Wave_object.function_space, name="pressure")
        for t in range(nt)
        if t % Wave_object.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    B = Wave_object.B
    rhs = Wave_object.rhs

    for step in range(nt):
        rhs_forcing.assign(0.0)
        B = fire.assemble(rhs, tensor=B)
        f = excitations.apply_source(rhs_forcing, Wave_object.wavelet[step])
        B0 = B.sub(0)
        B0 += f
        Wave_object.solver.solve()

        u_np1.assign(X)

        usol_recv.append(
            Wave_object.receivers.interpolate(u_np1.dat.data_ro_with_halos[:])
        )

        if step % Wave_object.gradient_sampling_frequency == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if (step - 1) % Wave_object.output_frequency == 0:
            assert (
                fire.norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the \
                mesh differently"
            if Wave_object.forward_output:
                output.write(u_n, time=t, name="Pressure")

            helpers.display_progress(Wave_object.comm, t)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    Wave_object.current_time = t
    helpers.display_progress(Wave_object.comm, t)

    usol_recv = helpers.fill(
        usol_recv, receivers.is_local, nt, receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, comm)
    Wave_object.receivers_output = usol_recv

    Wave_object.forward_solution = usol
    Wave_object.forward_solution_receivers = usol_recv

    return usol, usol_recv


def mixed_space_central_difference(Wave_object, source_id=0):
    """
    Performs central difference time integration for wave propagation.
    Solves for  a mixed space formulation, for function X. For correctly
    outputing pressure, order the mixed function space so that the space
    pressure lives in is first.

    Parameters:
    -----------
    Wave_object: Spyro object
        The Wave object containing the necessary data and parameters.
    source_id: int (optional)
        The ID of the source being propagated. Defaults to 0.

    Returns:
    --------
        tuple:
            A tuple containing the forward solution and the receiver output.
    """
    excitations = Wave_object.sources
    excitations.current_source = source_id
    receivers = Wave_object.receivers
    comm = Wave_object.comm
    temp_filename = Wave_object.forward_output_file
    filename, file_extension = temp_filename.split(".")
    output_filename = filename + "sn" + str(source_id) + "." + file_extension
    if Wave_object.forward_output:
        parallel_print(f"Saving output in: {output_filename}", Wave_object.comm)

    output = fire.File(output_filename, comm=comm.comm)
    comm.comm.barrier()

    final_time = Wave_object.final_time
    dt = Wave_object.dt
    t = Wave_object.current_time
    nt = int(final_time / dt) + 1  # number of timesteps

    X = Wave_object.X
    X_n = Wave_object.X_n
    X_nm1 = Wave_object.X_nm1

    rhs_forcing = fire.Function(Wave_object.function_space)
    usol = [
        fire.Function(Wave_object.function_space, name="pressure")
        for t in range(nt)
        if t % Wave_object.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    B = Wave_object.B
    rhs_ = Wave_object.rhs

    for step in range(nt):
        rhs_forcing.assign(0.0)
        B = fire.assemble(rhs_, tensor=B)
        f = excitations.apply_source(rhs_forcing, Wave_object.wavelet[step])
        B0 = B.sub(0)
        B0 += f
        Wave_object.solver.solve(X, B)

        X_np1 = X

        X_nm1.assign(X_n)
        X_n.assign(X_np1)

        usol_recv.append(
            Wave_object.receivers.interpolate(
                X_np1.dat.data_ro_with_halos[0][:]
            )
        )

        if step % Wave_object.gradient_sampling_frequency == 0:
            usol[save_step].assign(X_np1.sub(0))
            save_step += 1

        if (step - 1) % Wave_object.output_frequency == 0:
            assert (
                fire.norm(X_np1.sub(0)) < 1
            ), "Numerical instability. Try reducing dt or building the \
                mesh differently"
            if Wave_object.forward_output:
                output.write(X_np1.sub(0), time=t, name="Pressure")

            helpers.display_progress(comm, t)

        t = step * float(dt)

    Wave_object.current_time = t
    helpers.display_progress(Wave_object.comm, t)

    usol_recv = helpers.fill(
        usol_recv, receivers.is_local, nt, receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, comm)
    Wave_object.receivers_output = usol_recv

    Wave_object.forward_solution = usol
    Wave_object.forward_solution_receivers = usol_recv

    return usol, usol_recv


def central_difference_MMS(Wave_object, source_id=0):
    """Propagates the wave forward in time.
    Currently uses central differences.

    Parameters:
    -----------
    dt: Python 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the model_parameters.
    final_time: Python 'float' (optional)
        Time which simulation ends. If not mentioned uses the default,
        that was estabilished in the model_parameters.
    """
    receivers = Wave_object.receivers
    comm = Wave_object.comm
    temp_filename = Wave_object.forward_output_file
    filename, file_extension = temp_filename.split(".")
    output_filename = filename + "sn_mms_" + "." + file_extension
    if Wave_object.forward_output:
        print(f"Saving output in: {output_filename}", flush=True)

    output = fire.File(output_filename, comm=comm.comm)
    comm.comm.barrier()

    X = fire.Function(Wave_object.function_space)

    final_time = Wave_object.final_time
    dt = Wave_object.dt
    t = Wave_object.current_time
    nt = int((final_time - t) / dt) + 1  # number of timesteps

    u_nm1 = Wave_object.u_nm1
    u_n = Wave_object.u_n
    u_np1 = Wave_object.u_np1
    u_nm1.assign(Wave_object.analytical_solution(t - 2 * dt))
    u_n.assign(Wave_object.analytical_solution(t - dt))
    # u_np1 = fire.Function(Wave_object.function_space, name="pressure t +dt")
    # u_nm1.dat.data[:] = np.load("old_u_nm1.npy")
    # u_n.dat.data[:] = np.load("old_u_n.npy")
    u = fire.TrialFunction(Wave_object.function_space)
    v = fire.TestFunction(Wave_object.function_space)

    usol = [
        fire.Function(Wave_object.function_space, name="pressure")
        for t in range(nt)
        if t % Wave_object.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    B = Wave_object.B
    rhs = Wave_object.rhs
    quad_rule = Wave_object.quadrature_rule

    q_xy = Wave_object.q_xy

    for step in range(nt):
        q = q_xy * Wave_object.mms_source_in_time(t)
        m1 = (
            1
            / (Wave_object.c * Wave_object.c)
            * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dx(scheme=quad_rule)
        )
        a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
        le = q * v * dx(scheme=quad_rule)

        form = m1 + a - le
        rhs = fire.rhs(form)

        B = fire.assemble(rhs, tensor=B)

        Wave_object.solver.solve(u_np1, B)

        # u_np1.assign(X)

        usol_recv.append(
            Wave_object.receivers.interpolate(u_np1.dat.data_ro_with_halos[:])
        )

        if step % Wave_object.gradient_sampling_frequency == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if (step - 1) % Wave_object.output_frequency == 0:
            assert (
                fire.norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the \
                mesh differently"
            if Wave_object.forward_output:
                output.write(u_n, time=t, name="Pressure")
            if t > 0:
                helpers.display_progress(Wave_object.comm, t)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    Wave_object.current_time = t
    helpers.display_progress(Wave_object.comm, t)
    Wave_object.analytical_solution(t)

    usol_recv = helpers.fill(
        usol_recv, receivers.is_local, nt, receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, comm)
    Wave_object.receivers_output = usol_recv

    return usol, usol_recv
