import firedrake as fire
from . import helpers


def backward_wave_propagator(Wave_obj, dt=None):
    """Propagates the adjoint wave backwards in time.
    Currently uses central differences.

    Parameters:
    -----------
    Wave_obj: Spyro wave object
        Wave object that already propagated a forward wave.
    dt: Python 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.

    Returns:
    --------
    dJ: Firedrake 'Function'
        Calculated gradient
    """
    if Wave_obj.abc_active is False:
        return backward_wave_propagator_no_pml(Wave_obj, dt=dt)
    elif Wave_obj.abc_active:
        return mixed_space_backward_wave_propagator(Wave_obj, dt=dt)


def backward_wave_propagator_no_pml(Wave_obj, dt=None):
    """Propagates the adjoint wave backwards in time.
    Currently uses central differences. Does not have any PML.

    Parameters:
    -----------
    Wave_obj: Spyro wave object
        Wave object that already propagated a forward wave.
    dt: Python 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.

    Returns:
    --------
    dJ: Firedrake 'Function'
        Calculated gradient
    """
    Wave_obj.reset_pressure()
    if dt is not None:
        Wave_obj.dt = dt

    forward_solution = Wave_obj.forward_solution
    receivers = Wave_obj.receivers
    residual = Wave_obj.misfit
    comm = Wave_obj.comm
    temp_filename = Wave_obj.forward_output_filename

    filename, file_extension = temp_filename.split(".")
    output_filename = "backward." + file_extension

    output = fire.VTKFile(output_filename)
    comm.comm.barrier()

    X = fire.Function(Wave_obj.function_space)
    dJ = fire.Function(Wave_obj.function_space)  # , name="gradient")

    final_time = Wave_obj.final_time
    dt = Wave_obj.dt
    t = Wave_obj.current_time
    if t != final_time:
        print(f"Current time of {t}, different than final_time of {final_time}. Setting final_time to current time in backwards propagation.", flush=True)
    nt = int(t / dt) + 1  # number of timesteps

    u_nm1 = Wave_obj.u_nm1
    u_n = Wave_obj.u_n
    u_np1 = fire.Function(Wave_obj.function_space)

    rhs_forcing = fire.Cofunction(Wave_obj.function_space.dual())

    B = Wave_obj.B
    rhs = Wave_obj.rhs

    # Define a gradient problem
    m_u = fire.TrialFunction(Wave_obj.function_space)
    m_v = fire.TestFunction(Wave_obj.function_space)
    mgrad = m_u * m_v * fire.dx(scheme=Wave_obj.quadrature_rule)

    dufordt2 = fire.Function(Wave_obj.function_space)
    uadj = fire.Function(Wave_obj.function_space)  # auxiliarly function for the gradient compt.

    ffG = -2 * (Wave_obj.c)**(-3) * fire.dot(dufordt2, uadj) * m_v * fire.dx(scheme=Wave_obj.quadrature_rule)

    lhsG = mgrad
    rhsG = ffG

    gradi = fire.Function(Wave_obj.function_space)
    grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)
    grad_solver = fire.LinearVariationalSolver(
        grad_prob,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )

    # assembly_callable = create_assembly_callable(rhs, tensor=B)

    for step in range(nt-1, -1, -1):
        rhs_forcing.assign(0.0)
        B = fire.assemble(rhs, tensor=B)
        f = receivers.apply_receivers_as_source(rhs_forcing, residual, step)
        B0 = B.sub(0)
        B0 += f
        Wave_obj.solver.solve(X, B)

        u_np1.assign(X)

        if (step) % Wave_obj.output_frequency == 0:
            assert (
                fire.norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the \
                mesh differently"
            if Wave_obj.forward_output:
                output.write(u_n, time=t, name="Pressure")

            helpers.display_progress(Wave_obj.comm, t)

        if step % Wave_obj.gradient_sampling_frequency == 0:
            # duadjdt2.assign( ((u_np1 - 2.0 * u_n + u_nm1) / fire.Constant(dt**2)) )
            uadj.assign(u_np1)
            if len(forward_solution) > 2:
                dufordt2.assign(
                    (forward_solution.pop() - 2.0 * forward_solution[-1] + forward_solution[-2]) / fire.Constant(dt**2)
                )
            else:
                dufordt2.assign(
                    (forward_solution.pop() - 2.0 * 0.0 + 0.0) / fire.Constant(dt**2)
                )

            grad_solver.solve()
            if step == nt-1 or step == 0:
                dJ += gradi
            else:
                dJ += 2*gradi

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    Wave_obj.current_time = t
    helpers.display_progress(Wave_obj.comm, t)

    dJ.dat.data_with_halos[:] *= (dt/2)
    return dJ


def mixed_space_backward_wave_propagator(Wave_obj, dt=None):
    """Propagates the adjoint wave backwards in time.
    Currently uses central differences. Based on the
    mixed space implementation of PML.

    Parameters:
    -----------
    Wave_obj: Spyro wave object
        Wave object that already propagated a forward wave.
    dt: Python 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.

    Returns:
    --------
    dJ: Firedrake 'Function'
        Calculated gradient
    """
    Wave_obj.reset_pressure()
    if dt is not None:
        Wave_obj.dt = dt

    forward_solution = Wave_obj.forward_solution
    receivers = Wave_obj.receivers
    residual = Wave_obj.misfit
    comm = Wave_obj.comm
    temp_filename = Wave_obj.forward_output_filename

    filename, file_extension = temp_filename.split(".")
    output_filename = "backward." + file_extension

    output = fire.VTKFile(output_filename)
    comm.comm.barrier()

    X = Wave_obj.X
    dJ = fire.Function(Wave_obj.function_space)  # , name="gradient")

    final_time = Wave_obj.final_time
    dt = Wave_obj.dt
    t = Wave_obj.current_time
    if t != final_time:
        print(f"Current time of {t}, different than final_time of {final_time}. Setting final_time to current time in backwards propagation.", flush=True)
    nt = int(t / dt) + 1  # number of timesteps

    X_nm1 = Wave_obj.X_nm1
    X_n = Wave_obj.X_n
    X_np1 = fire.Function(Wave_obj.mixed_function_space)

    rhs_forcing = fire.Cofunction(Wave_obj.function_space.dual())

    B = Wave_obj.B
    rhs = Wave_obj.rhs

    # Define a gradient problem
    m_u = fire.TrialFunction(Wave_obj.function_space)
    m_v = fire.TestFunction(Wave_obj.function_space)
    mgrad = m_u * m_v * fire.dx(scheme=Wave_obj.quadrature_rule)

    # dufordt2 = fire.Function(Wave_obj.function_space)
    ufor = fire.Function(Wave_obj.function_space)
    uadj = fire.Function(Wave_obj.function_space)  # auxiliarly function for the gradient compt.

    # ffG = -2 * (Wave_obj.c)**(-3) * fire.dot(dufordt2, uadj) * m_v * fire.dx(scheme=Wave_obj.quadrature_rule)
    ffG = 2.0 * Wave_obj.c * fire.dot(fire.grad(uadj), fire.grad(ufor)) * m_v * fire.dx(scheme=Wave_obj.quadrature_rule)

    lhsG = mgrad
    rhsG = ffG

    gradi = fire.Function(Wave_obj.function_space)
    grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)
    grad_solver = fire.LinearVariationalSolver(
        grad_prob,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )

    # assembly_callable = create_assembly_callable(rhs, tensor=B)

    for step in range(nt-1, -1, -1):
        rhs_forcing.assign(0.0)
        B = fire.assemble(rhs, tensor=B)
        f = receivers.apply_receivers_as_source(rhs_forcing, residual, step)
        B0 = B.sub(0)
        B0 += f
        Wave_obj.solver.solve(X, B)

        X_np1.assign(X)

        if (step) % Wave_obj.output_frequency == 0:
            if Wave_obj.forward_output:
                output.write(X_n.sub(0), time=t, name="Pressure")

            helpers.display_progress(Wave_obj.comm, t)

        if step % Wave_obj.gradient_sampling_frequency == 0:
            # duadjdt2.assign( ((u_np1 - 2.0 * u_n + u_nm1) / fire.Constant(dt**2)) )
            uadj.assign(X_np1.sub(0))
            ufor.assign(forward_solution.pop())

            grad_solver.solve()
            if step == nt-1 or step == 0:
                dJ += gradi
            else:
                dJ += 2*gradi

        X_nm1.assign(X_n)
        X_n.assign(X_np1)

        t = step * float(dt)

    Wave_obj.current_time = t
    helpers.display_progress(Wave_obj.comm, t)

    dJ.dat.data_with_halos[:] *= (dt/2)
    return dJ
