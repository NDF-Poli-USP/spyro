import firedrake as fire
from . import helpers
from .wave import Wave
from ..io.basicio import parallel_print


def backward_wave_propagator(wave_obj: Wave, dt: float = None) -> fire.Function:
    """Propagates the adjoint wave backwards in time.

    Currently uses central differences.

    Parameters:
    -----------
    wave_obj : Wave
        Wave object that already propagated a forward wave.
    dt : float (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.

    Returns:
    --------
    dJ : Firedrake 'Function'
        Calculated gradient

    Notes:
    ------
    This is an unified backward wave propagation for both PML and no-PML cases.
    The PML path uses the mixed-space gradient form ``2c * ∇u_adj · ∇u_fwd``
    while the no-PML path uses ``-2/c³ * ü_fwd * u_adj``.
    Source injection uses ``wave_obj.rhs_no_pml_source()`` and the prebuilt
    variational solver is advanced with ``wave_obj.solver.solve()``.
    """
    wave_obj.reset_pressure()
    mask_available = wave_obj.gradient_mask_available
    if dt is not None:
        wave_obj.dt = dt

    dt = wave_obj.dt
    t = wave_obj.current_time
    if t != wave_obj.final_time:
        parallel_print(
            f"Current time of {t}, different than final_time of "
            f"{wave_obj.final_time}. Setting final_time to current time "
            f"in backwards propagation.", wave_obj.comm,
        )
    nt = int(t / dt) + 1

    wave_obj.comm.comm.barrier()

    gradient_space = wave_obj.get_scalar_function_space()
    dJ = fire.Function(gradient_space)
    rhs_forcing = fire.Cofunction(gradient_space.dual())

    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        wave_obj, mask_available,
    )

    forward_solution = wave_obj.forward_solution
    receivers = wave_obj.receivers
    residual = wave_obj.misfit

    for step in range(nt - 1, -1, -1):
        rhs_forcing.assign(0.0)
        wave_obj.rhs_no_pml_source().assign(
            receivers.apply_receivers_as_source(rhs_forcing, residual, step)
        )
        wave_obj.solver.solve()

        if step % wave_obj.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(wave_obj.get_function(state=wave_obj.next_vstate))

            if wave_obj.abc_boundary_layer_type == "PML":
                # Pop to keep the list in sync, but use the element one
                # step behind so that u_fwd and u_adj are at the same
                # physical time (usol[k] = u^{k+1}; we need u^k).
                forward_solution.pop()
                if len(forward_solution) > 0:
                    forward_field.assign(forward_solution[-1])
                else:
                    forward_field.assign(0.0)
            else:
                forward_field.assign(_compute_dufordt2(forward_solution, dt))
            grad_solver.solve()
            _trapezoidal_gradient_integration(dJ, gradi, step, nt)

        if wave_obj.abc_boundary_layer_type == "PML":
            wave_obj.X_nm1.assign(wave_obj.X_n)
            wave_obj.X_n.assign(wave_obj.X_np1)
        else:
            wave_obj.u_nm1.assign(wave_obj.u_n)
            wave_obj.u_n.assign(wave_obj.u_np1)
        t = step * float(dt)

    wave_obj.adjoint_solution = uadj
    wave_obj.current_time = t

    helpers.display_progress(wave_obj.comm, t)

    dJ.dat.data_with_halos[:] *= dt / 2
    return dJ


def _pml_interior_indicator(wave_obj: Wave) -> fire.conditional:
    """UFL indicator: 1 inside the physical domain, 0 in the PML layer."""
    # TODO: This is a bit hacky, will be not needed when submeshes are enabled in Spyro.
    z = wave_obj.mesh_z
    x = wave_obj.mesh_x
    z_min = -(wave_obj.mesh_parameters.length_z)
    x_min = 0.0
    x_max = wave_obj.mesh_parameters.length_x

    inside = fire.And(fire.And(z >= z_min, x >= x_min), x <= x_max)

    if wave_obj.dimension == 3:
        y = wave_obj.mesh_y
        y_min = 0.0
        y_max = wave_obj.mesh_parameters.length_y
        inside = fire.And(inside, fire.And(y >= y_min, y <= y_max))

    return fire.conditional(inside, 1.0, 0.0)


def _build_gradient_solver(wave_obj: Wave, mask_available: bool) -> tuple[
        fire.LinearVariationalSolver, fire.Function, fire.Function, fire.Function
]:
    """Assemble the gradient variational problem.

    Parameters:
    -----------
    wave_obj : Wave
        The wave object containing the forward and adjoint solutions, as well
        as the velocity model and other parameters needed to build the
        gradient problem.
    mask_available : bool
        Flag indicating whether a gradient mask is available. If True, the
        gradient will be computed only in the inner region of the domain.

    Returns:
    --------
    grad_solver, forward_field, uadj, gradi
    """
    V = wave_obj.get_scalar_function_space()
    qr = wave_obj.quadrature_rule

    m_u = fire.TrialFunction(V)
    m_v = fire.TestFunction(V)
    if mask_available:
        # Use masked integration over inner region only
        dx = fire.dx(2, scheme=qr)
        mask_available = True
    else:
        dx = fire.dx(**qr)
        mask_available = False

    mgrad = m_u * m_v * dx
    forward_field = fire.Function(V)
    uadj = fire.Function(V)

    if wave_obj.abc_boundary_layer_type == "PML":
        # Always exclude PML region from gradient.
        # This is necessary once the gradient expression is not considering
        # the PML auxiliary variables. In addition, we are not interested
        # in the gradient in the PML region.
        indicator = _pml_interior_indicator(wave_obj)
        # Computes de gradient only in the physical domain.
        ffG = (
            2.0 * wave_obj.c * indicator * fire.dot(
                fire.grad(uadj), fire.grad(forward_field)) * m_v * dx
        )
        parallel_print(
            "Excluding PML region from gradient (mixed space)", wave_obj.comm
        )
    else:
        ffG = (
            -2 * (wave_obj.c) ** (-3) * fire.dot(forward_field, uadj) * m_v * dx
        )
        parallel_print(
            "No gradient mask found: computing gradients over full domain",
            wave_obj.comm,
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


def _compute_dufordt2(forward_solution: list, dt: float) -> fire.Function:
    """Second time-derivative via 3-point central finite differences."""
    if len(forward_solution) > 2:
        return (
            forward_solution.pop()
            - 2.0 * forward_solution[-1]
            + forward_solution[-2]
        ) / fire.Constant(dt**2)
    else:
        return forward_solution.pop() / fire.Constant(dt**2)


def _trapezoidal_gradient_integration(
        dJ: fire.Function, gradi: fire.Function, step: int, nt: int) -> None:
    """Trapezoidal-rule gradient accumulation.

    Parameters:
    -----------
    dJ : Firedrake 'Function'
        The accumulated gradient.
    gradi : Firedrake 'Function'
        The gradient at the current time step.
    step : int
        The current time step.
    nt : int
        The total number of time steps.
    """

    if step == nt - 1 or step == 0:
        dJ += gradi
    else:
        dJ += 2 * gradi
