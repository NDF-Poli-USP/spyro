import firedrake as fire
from . import helpers


def backward_wave_propagator(Wave_obj, dt=None):
    """Propagates the adjoint wave backwards in time.
    Currently uses central differences.

    Parameters:
    -----------
    Wave_obj: Spyro wave object
        Wave object that already propagated a forward wave.
    dt: 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object.

    Returns:
    --------
    dJ: Firedrake 'Function'
        Computed gradient of the cost functional with respect to the control variable.
    """
    if Wave_obj.abc_active:
        return _backward_propagation(Wave_obj, dt=dt, pml=True)
    else:
        return _backward_propagation(Wave_obj, dt=dt, pml=False)


def _build_gradient_solver(Wave_obj, mask_available, pml):
    """Assemble the gradient variational problem.

    Returns:
    --------
    grad_solver, forward_field, uadj, gradi
    """
    V = Wave_obj.get_scalar_function_space()
    qr = Wave_obj.quadrature_rule

    m_u = fire.TrialFunction(V)
    m_v = fire.TestFunction(V)
    if mask_available:
        # Use masked integration over inner region only
        mgrad = m_u * m_v * fire.dx(2, scheme=qr)
        mask_available = True
    else:
        # Fall back to full domain
        mgrad = m_u * m_v * fire.dx(**qr)
        mask_available = False

    forward_field = fire.Function(V)
    uadj = fire.Function(V)

    if pml and mask_available:
        ffG = (
            2.0
            * Wave_obj.c
            * fire.dot(fire.grad(uadj), fire.grad(forward_field))
            * m_v
            * fire.dx(2, scheme=qr)
        )
        if Wave_obj.comm.comm.rank == 0:
            print(
                "Applying gradient mask: gradients will be computed only "
                "in inside region (mixed space)",
                flush=True,
            )
    elif pml:
        ffG = (
            2.0
            * Wave_obj.c
            * fire.dot(fire.grad(uadj), fire.grad(forward_field))
            * m_v
            * fire.dx(**qr)
        )
        if Wave_obj.comm.comm.rank == 0:
            print(
                "No gradient mask found: computing gradients over full "
                "domain (mixed space)",
                flush=True,
            )
    elif mask_available:
        ffG = (
            -2
            * (Wave_obj.c) ** (-3)
            * fire.dot(forward_field, uadj)
            * m_v
            * fire.dx(2, scheme=qr)
        )
        if Wave_obj.comm.comm.rank == 0:
            print(
                "Applying gradient mask: gradients will be computed only "
                "in inside region",
                flush=True,
            )
    else:
        ffG = (
            -2
            * (Wave_obj.c) ** (-3)
            * fire.dot(forward_field, uadj)
            * m_v
            * fire.dx(**qr)
        )
        if Wave_obj.comm.comm.rank == 0:
            print(
                "No gradient mask found: computing gradients over full "
                "domain",
                flush=True,
            )

    gradi = fire.Function(V)
    grad_prob = fire.LinearVariationalProblem(mgrad, ffG, gradi)
    grad_solver = fire.LinearVariationalSolver(
        grad_prob,
        solver_parameters={
            "ksp_type": "preonly", "pc_type": "jacobi", "mat_type": "matfree",
        },
    )

    return grad_solver, forward_field, uadj, gradi


def _compute_dufordt2(forward_solution, dt):
    """Second time-derivative via 3-point central finite differences."""
    if len(forward_solution) > 2:
        return (
            forward_solution.pop()
            - 2.0 * forward_solution[-1]
            + forward_solution[-2]
        ) / fire.Constant(dt**2)
    else:
        return forward_solution.pop() / fire.Constant(dt**2)


def _trapezoidal_gradient_integration(dJ, gradi, step, nt):
    """Trapezoidal-rule gradient accumulation."""
    if step == nt - 1 or step == 0:
        dJ += gradi
    else:
        dJ += 2 * gradi


def _backward_propagation(Wave_obj, dt=None, pml=False):
    """Unified backward wave propagation for both PML and no-PML cases.

    Parameters:
    -----------
    Wave_obj: Spyro wave object
        Wave object that already propagated a forward wave.
    dt: 'float' (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.
    pml: bool
        Whether PML absorbing boundary conditions are active.

    Returns:
    --------
    dJ: Firedrake 'Function'
        Calculated gradient

    Notes:
    ------
    The PML path uses the mixed-space gradient form
    ``2c * ∇u_adj · ∇u_fwd`` while the no-PML path uses
    ``-2/c³ * ü_fwd * u_adj``.
    Source injection uses ``Wave_obj.rhs_no_pml_source()`` and the prebuilt
    variational solver is advanced with ``Wave_obj.solver.solve()``.
    """
    Wave_obj.reset_pressure()
    mask_available = Wave_obj.gradient_mask_available
    if dt is not None:
        Wave_obj.dt = dt

    dt = Wave_obj.dt
    t = Wave_obj.current_time
    if t != Wave_obj.final_time:
        print(
            f"Current time of {t}, different than final_time of "
            f"{Wave_obj.final_time}. Setting final_time to current time "
            f"in backwards propagation.",
            flush=True,
        )
    nt = int(t / dt) + 1

    Wave_obj.comm.comm.barrier()

    gradient_space = Wave_obj.get_scalar_function_space()
    dJ = fire.Function(gradient_space)
    rhs_forcing = fire.Cofunction(gradient_space.dual())

    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        Wave_obj, mask_available, pml,
    )

    forward_solution = Wave_obj.forward_solution
    receivers = Wave_obj.receivers
    residual = Wave_obj.misfit

    output = None
    output = _create_adjoint_output(Wave_obj)

    for step in range(nt - 1, -1, -1):
        rhs_forcing.assign(0.0)
        Wave_obj.rhs_no_pml_source().assign(
            receivers.apply_receivers_as_source(rhs_forcing, residual, step)
        )
        Wave_obj.solver.solve()

        if step % Wave_obj.output_frequency == 0:
            _output_step(Wave_obj, t, pml, output=output if pml else None)

        if step % Wave_obj.gradient_sampling_frequency == 0:
            uadj.assign(Wave_obj.get_wave_equation_state(Wave_obj.next_vstate))

            if pml:
                forward_field.assign(forward_solution.pop())
            else:
                forward_field.assign(_compute_dufordt2(forward_solution, dt))
            grad_solver.solve()
            _trapezoidal_gradient_integration(dJ, gradi, step, nt)

        if pml:
            Wave_obj.X_nm1.assign(Wave_obj.X_n)
            Wave_obj.X_n.assign(Wave_obj.X_np1)
        else:
            Wave_obj.u_nm1.assign(Wave_obj.u_n)
            Wave_obj.u_n.assign(Wave_obj.u_np1)
        t = step * float(dt)

    Wave_obj.current_time = t
    helpers.display_progress(Wave_obj.comm, t)

    dJ.dat.data_with_halos[:] *= dt / 2
    return dJ


def _create_adjoint_output(Wave_obj):
    """Create VTK output file for adjoint propagation."""
    temp_filename = Wave_obj.forward_output_filename
    _, file_extension = temp_filename.split(".")
    return fire.VTKFile("adjoint." + file_extension)


def _output_step(Wave_obj, t, pml, output=None):
    """Handle per-step output and stability checks."""
    if Wave_obj.forward_output:
        output.write(Wave_obj.get_function(), time=t, name="Pressure")
    else:
        assert (
            fire.norm(Wave_obj.u_n) < 1
        ), "Numerical instability. Try reducing dt or building the mesh differently"

    helpers.display_progress(Wave_obj.comm, t)
