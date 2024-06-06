import firedrake as fire
from firedrake import Constant, dx, dot, grad
from firedrake.assemble import create_assembly_callable
import numpy as np

from ..io.basicio import parallel_print
from . import helpers
from .. import utils
# from ..habc import lenCam_spy


def central_difference(Wave_object, source_id=0):
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

    rhs_forcing = fire.Function(Wave_object.function_space)
    usol = [
        fire.Function(Wave_object.function_space, name="pressure")
        for t in range(nt)
        if t % Wave_object.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    B = Wave_object.B
    rhs = Wave_object.rhs

    check_boundary = True

    # assembly_callable = create_assembly_callable(rhs, tensor=B)
    if check_boundary is True:
        function_z = fire.Function(X)
        function_z.interpolate(Wave_object.mesh_z)
        function_x = fire.Function(X)
        function_x.interpolate(Wave_object.mesh_x)
        tol = 1e-6
        left_boundary = np.where(function_x.dat.data[:] <= tol)
        right_boundary = np.where(function_x.dat.data[:] >= Wave_object.length_x-tol)
        bottom_boundary = np.where(function_z.dat.data[:] <= tol-Wave_object.length_z)
        pressure_on_left = u_n.dat.data_ro_with_halos[left_boundary]
        pressure_on_right = u_n.dat.data_ro_with_halos[right_boundary]
        pressure_on_bottom = u_n.dat.data_ro_with_halos[bottom_boundary]
        threshold = 1e-6
        check_left = True
        check_right = True
        check_bottom = True
        t_left = np.inf
        t_right = np.inf
        t_bottom = np.inf
        bottom_point_dof = np.nan
        left_point_dof = np.nan
        right_point_dof = np.nan

    for step in range(nt):
        rhs_forcing.assign(0.0)
        B = fire.assemble(rhs, tensor=B)
        f = excitations.apply_source(rhs_forcing, Wave_object.wavelet[step])
        B0 = B.sub(0)
        B0 += f
        Wave_object.solver.solve(X, B)

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

        pressure_on_left = u_n.dat.data_ro_with_halos[left_boundary]
        pressure_on_right = u_n.dat.data_ro_with_halos[right_boundary]
        pressure_on_bottom = u_n.dat.data_ro_with_halos[bottom_boundary]

        if np.any(np.abs(pressure_on_left) > threshold) and check_left:
            print("Pressure on left boundary is not zero")
            print(f"Time hit left boundary = {t}")
            t_left = t
            check_left = False
            vector_indices = np.where(np.abs(pressure_on_left) > threshold)
            vector_indices = vector_indices[0]
            global_indices = left_boundary[0][vector_indices]
            z_values = function_z.dat.data[global_indices]
            z_avg = np.average(z_values)
            indice = global_indices[np.argmin(np.abs(z_values-z_avg))]
            left_point_dof = indice

        if np.any(np.abs(pressure_on_right) > threshold) and check_right:
            print("Pressure on right boundary is not zero")
            print(f"Time hit right boundary = {t}")
            t_right = t
            check_right = False
            vector_indices = np.where(np.abs(pressure_on_right) > threshold)
            vector_indices = vector_indices[0]
            global_indices = right_boundary[0][vector_indices]
            z_values = function_z.dat.data[global_indices]
            z_avg = np.average(z_values)
            indice = global_indices[np.argmin(np.abs(z_values-z_avg))]
            right_point_dof = indice

        if np.any(np.abs(pressure_on_bottom) > threshold) and check_bottom:
            print("Pressure on bottom boundary is not zero")
            print(f"Time hit bottom boundary = {t}")
            t_bottom = t
            check_bottom = False
            vector_indices = np.where(np.abs(pressure_on_bottom) > threshold)
            vector_indices = vector_indices[0]
            global_indices = bottom_boundary[0][vector_indices]
            x_values = function_x.dat.data[global_indices]
            x_avg = np.average(x_values)
            indice = global_indices[np.argmin(np.abs(x_values-x_avg))]
            bottom_point_dof = indice

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    Wave_object.current_time = t
    noneikonal_dofs = [left_point_dof, right_point_dof, bottom_point_dof]
    Wave_object.neik_time_value = np.min([t_left, t_right, t_bottom])
    Wave_object.noneikonal_dof = noneikonal_dofs[np.argmin([t_left, t_right, t_bottom])]
    Wave_object.neik_location = (function_z.dat.data[Wave_object.noneikonal_dof], function_x.dat.data[Wave_object.noneikonal_dof])
    Wave_object.neik_velocity_value = Wave_object.c.dat.data_ro_with_halos[Wave_object.noneikonal_dof]
    # Wave_object.noneikonal_minimum_point = eikonal_points[]
    helpers.display_progress(Wave_object.comm, t)
    diameters = fire.CellDiameter(Wave_object.mesh)
    Wave_object.h_min = fire.assemble(diameters * fire.dx)

    # fref, F_L, pad_length, lref = lenCam_spy.habc_size(Wave_object)
    # print(f"L ref = {lref}")
    usol_recv = helpers.fill(
        usol_recv, receivers.is_local, nt, receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, comm)
    Wave_object.receivers_output = usol_recv

    Wave_object.forward_solution = usol
    Wave_object.forward_solution_receivers = usol_recv

    return usol, usol_recv


def mixed_space_central_difference(Wave_object, source_id=0):
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

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    for step in range(nt):
        rhs_forcing.assign(0.0)
        assembly_callable()
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
    u_nm1.assign(Wave_object.analytical_solution(t - 2 * dt))
    u_n.assign(Wave_object.analytical_solution(t - dt))
    u_np1 = fire.Function(Wave_object.function_space, name="pressure t +dt")
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

    # assembly_callable = create_assembly_callable(rhs, tensor=B)
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

        Wave_object.solver.solve(X, B)

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


