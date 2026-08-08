import firedrake as fire
from . import helpers
from .wave import Wave
from ..io.basicio import parallel_print
from ..receivers.Receivers import Receivers
from ..utils.typing import AbsorbingBCsType


def backward_wave_propagator(wave: Wave, dt: float = None) -> fire.Function:
    """Propagates the adjoint wave backwards in time.

    Currently uses central differences.

    Parameters:
    -----------
    wave : Wave
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
    Source injection uses ``wave.rhs_no_pml_source()`` and the prebuilt
    variational solver is advanced with ``wave.solver.solve()``.
    """
    wave.reset_pressure()
    mask_available = wave.gradient_mask_available
    if dt is not None:
        wave.dt = dt

    dt = wave.dt
    t = wave.current_time
    if t != wave.final_time:
        parallel_print(
            f"Current time of {t}, different than final_time of "
            f"{wave.final_time}. Setting final_time to current time "
            f"in backwards propagation.", wave.comm,
        )
    nt = int(t / dt) + 1

    # The forward wavefield is stored only every ``gradient_sampling_frequency``
    # steps, so consecutive stored samples are ``sample_dt = freq * dt`` apart in
    # physical time. Every time-derivative stencil and every quadrature weight in
    # the gradient must use ``sample_dt`` (not ``dt``); otherwise the gradient is
    # off by a factor of ``freq**2`` (second derivative) and ``freq``
    # (trapezoidal spacing). ``last_sample`` is the largest sampled step index,
    # i.e. the last endpoint of the trapezoidal rule over sampled steps.
    freq = wave.gradient_sampling_frequency
    sample_dt = freq * dt
    last_sample = ((nt - 1) // freq) * freq

    wave.comm.comm.barrier()

    gradient_space = wave.get_scalar_function_space()
    dJ = fire.Function(gradient_space)
    rhs_forcing = fire.Cofunction(gradient_space.dual())

    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        wave, mask_available,
    )

    forward_solution = wave.forward_solution
    receivers = wave.receivers

    for step in range(nt - 1, -1, -1):
        rhs_forcing.assign(0.0)
        receiver_source = receivers.apply_receivers_as_source(
            rhs_forcing, wave.misfit, step,
        )
        if step == 0 or step == nt - 1:
            receiver_source.assign(0.5 * receiver_source)
        wave.rhs_no_pml_source().assign(
            receiver_source
        )
        wave.solver.solve()

        if step % wave.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(wave.get_function(state=wave.next_vstate))

            if wave.abc_type == AbsorbingBCsType.PML:
                # Pop to keep the list in sync, but use the element one
                # step behind so that u_fwd and u_adj are at the same
                # physical time (usol[k] = u^{k+1}; we need u^k).
                forward_solution.pop()
                if len(forward_solution) > 0:
                    forward_field.assign(forward_solution[-1])
                else:
                    forward_field.assign(0.0)
            else:
                forward_field.assign(_compute_dufordt2(forward_solution, sample_dt))
            grad_solver.solve()
            _trapezoidal_gradient_integration(dJ, gradi, step, last_sample)

        if wave.abc_type == AbsorbingBCsType.PML:
            wave.X_nm1.assign(wave.X_n)
            wave.X_n.assign(wave.X_np1)
        else:
            wave.u_nm1.assign(wave.u_n)
            wave.u_n.assign(wave.u_np1)
        t = step * float(dt)

    wave.adjoint_solution = uadj
    wave.current_time = t

    helpers.display_progress(wave.comm, t)

    dJ.dat.data_with_halos[:] *= sample_dt / 2
    return dJ


def _pml_interior_indicator(wave: Wave) -> fire.conditional:
    """UFL indicator: 1 inside the physical domain, 0 in the PML layer."""
    # TODO: This is a bit hacky, will be not needed when submeshes are enabled in Spyro.
    z = wave.mesh_z
    x = wave.mesh_x
    z_min = -(wave.mesh_parameters.length_z)
    x_min = 0.0
    x_max = wave.mesh_parameters.length_x

    inside = fire.And(fire.And(z >= z_min, x >= x_min), x <= x_max)

    if wave.dimension == 3:
        y = wave.mesh_y
        y_min = 0.0
        y_max = wave.mesh_parameters.length_y
        inside = fire.And(inside, fire.And(y >= y_min, y <= y_max))

    return fire.conditional(inside, 1.0, 0.0)


def _build_gradient_solver(wave: Wave, mask_available: bool) -> tuple[
        fire.LinearVariationalSolver, fire.Function, fire.Function, fire.Function
]:
    """Assemble the gradient variational problem.

    Parameters:
    -----------
    wave : Wave
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
    if wave.use_vertex_only_mesh and wave.automatic_adjoint is False:
        # WARNING: Mega ultra gambiarra
        # TODO: open issue and fix this in another PR
        wave.use_vertex_only_mesh = False
        wave.receivers = Receivers(wave)
        wave.use_vertex_only_mesh = True
    V = wave.get_scalar_function_space()
    qr = wave.quadrature_rule

    m_u = fire.TrialFunction(V)
    m_v = fire.TestFunction(V)
    if mask_available:
        # Use masked integration over inner region only
        dx = fire.dx(2, **qr)
        mask_available = True
    else:
        dx = fire.dx(**qr)
        mask_available = False

    mgrad = m_u * m_v * dx
    forward_field = fire.Function(V)
    uadj = fire.Function(V)

    if wave.abc_type == AbsorbingBCsType.PML:
        # Always exclude PML region from gradient.
        # This is necessary once the gradient expression is not considering
        # the PML auxiliary variables. In addition, we are not interested
        # in the gradient in the PML region.
        indicator = _pml_interior_indicator(wave)
        # Compute the gradient only in the physical domain.

        """
        TODO: Refactor the gradient due to new PML formulation
        TODO: Add citations
        Formulation based on:
           "Efficient PML for the wave equation". Grote and Sim (2010)
           "A Modified PML Acoustic Wave Equation". Kim (2019)
        Acoustic Eq. is modified by dividing by c^2 (see implementation).
        The remaining PML Eqs. remanin unchanged.
        """

        ffG = (
            2.0 * wave.velocity_model * indicator * fire.dot(
                fire.grad(uadj), fire.grad(forward_field)) * m_v * dx
        )
        raise ValueError("PML gradient calculation temporarily unavailable")

    else:
        ffG = (
            -2 * (wave.velocity_model) ** (-3) * fire.dot(forward_field, uadj) * m_v * dx
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


def _compute_dufordt2(forward_solution: list, sample_dt: float) -> fire.Function:
    """Second time-derivative via 3-point finite differences.

    ``sample_dt`` is the physical time between consecutive stored samples
    (``gradient_sampling_frequency * dt``), which equals ``dt`` only when every
    step is stored.
    """
    if len(forward_solution) > 2:
        return (
            forward_solution.pop()
            - 2.0 * forward_solution[-1]
            + forward_solution[-2]
        ) / fire.Constant(sample_dt**2)
    else:
        return forward_solution.pop() / fire.Constant(sample_dt**2)


def _trapezoidal_gradient_integration(
        dJ: fire.Function, gradi: fire.Function, step: int,
        last_sample: int) -> None:
    """Trapezoidal-rule gradient accumulation over the stored (sampled) steps.

    Parameters:
    -----------
    dJ : Firedrake 'Function'
        The accumulated gradient.
    gradi : Firedrake 'Function'
        The gradient at the current time step.
    step : int
        The current time step.
    last_sample : int
        The largest sampled step index. Together with step 0 these are the two
        endpoints of the trapezoidal rule (weight 1); interior samples weight 2.
    """

    if step == last_sample or step == 0:
        dJ += gradi
    else:
        dJ += 2 * gradi
