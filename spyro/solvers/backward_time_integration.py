import firedrake as fire
from . import helpers
from .wave import Wave
from ..io.basicio import parallel_print


def backward_wave_propagator(Wave_obj: Wave, dt: float = None) -> fire.Function:
    """Propagates the adjoint wave backwards in time.

    Currently uses central differences.

    Parameters:
    -----------
    Wave_obj : Wave
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
        parallel_print(
            f"Current time of {t}, different than final_time of "
            f"{Wave_obj.final_time}. Setting final_time to current time "
            f"in backwards propagation.", Wave_obj.comm,
        )
    nt = int(t / dt) + 1

    Wave_obj.comm.comm.barrier()

    gradient_space = Wave_obj.get_scalar_function_space()
    dJ = fire.Function(gradient_space)
    rhs_forcing = fire.Cofunction(gradient_space.dual())

    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        Wave_obj, mask_available,
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
            _output_step(
                Wave_obj, t, output=output if Wave_obj.abc_boundary_layer_type == "PML" else None
            )

        if step % Wave_obj.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(Wave_obj.get_function(state=Wave_obj.next_vstate))

            if Wave_obj.abc_boundary_layer_type == "PML":
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

        if Wave_obj.abc_boundary_layer_type == "PML":
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


def _pml_interior_indicator(Wave_obj: Wave) -> fire.UFL.conditional:
    """UFL indicator: 1 inside the physical domain, 0 in the PML layer."""
    # TODO: This is a bit hacky, will be not needed when submeshes are enabled in Spyro.
    z = Wave_obj.mesh_z
    x = Wave_obj.mesh_x
    z_min = -(Wave_obj.mesh_parameters.length_z)
    x_min = 0.0
    x_max = Wave_obj.mesh_parameters.length_x

    inside = fire.And(fire.And(z >= z_min, x >= x_min), x <= x_max)

    if Wave_obj.dimension == 3:
        y = Wave_obj.mesh_y
        y_min = 0.0
        y_max = Wave_obj.mesh_parameters.length_y
        inside = fire.And(inside, fire.And(y >= y_min, y <= y_max))

    return fire.conditional(inside, 1.0, 0.0)


def _build_gradient_solver(Wave_obj: Wave, mask_available: bool) -> tuple[
        fire.LinearVariationalSolver, fire.Function, fire.Function, fire.Function
]:
    """Assemble the gradient variational problem.

    Parameters:
    -----------
    Wave_obj : Wave
        The wave object containing the forward and adjoint solutions, as well as the
        velocity model and other parameters needed to build the gradient problem.
    mask_available : bool
        Flag indicating whether a gradient mask is available. If True, the gradient
        will be computed only in the inner region of the domain.

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
        dx = fire.dx(2, scheme=qr)
        mask_available = True
    else:
        dx = fire.dx(**qr)
        mask_available = False

    mgrad = m_u * m_v * dx
    forward_field = fire.Function(V)
    uadj = fire.Function(V)

    if Wave_obj.abc_boundary_layer_type == "PML":
        # Always exclude PML region from gradient.
        # This is necessary once the gradient expression is not considering
        # the PML auxiliary variables. In addition, we are not interested
        # in the gradient in the PML region.
        indicator = _pml_interior_indicator(Wave_obj)
        # Computes de gradient only in the physical domain.
        ffG = (
            2.0 * Wave_obj.c * indicator * fire.dot(
                fire.grad(uadj), fire.grad(forward_field)) * m_v * dx
        )
        parallel_print(
            "Excluding PML region from gradient (mixed space)", Wave_obj.comm
        )
    else:
        ffG = (
            -2 * (Wave_obj.c) ** (-3) * fire.dot(forward_field, uadj) * m_v * dx
        )
        parallel_print(
            "No gradient mask found: computing gradients over full domain",
            Wave_obj.comm,
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


def _create_adjoint_output(Wave_obj: Wave) -> fire.VTKFile:
    """Create VTK output file for adjoint propagation."""
    temp_filename = Wave_obj.forward_output_filename
    _, file_extension = temp_filename.split(".")
    return fire.VTKFile("adjoint." + file_extension)


def _output_step(Wave_obj: Wave, t: float, output: fire.VTKFile = None) -> None:
    """Handle per-step output and stability checks."""
    if Wave_obj.forward_output:
        output.write(Wave_obj.get_function(), time=t, name="Pressure")
    else:
        assert (
            fire.norm(Wave_obj.u_n) < 1
        ), "Numerical instability. Try reducing dt or building the mesh differently"

    helpers.display_progress(Wave_obj.comm, t)
