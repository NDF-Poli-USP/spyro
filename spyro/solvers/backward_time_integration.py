import firedrake as fire
from . import helpers
from .wave import Wave
from ..io.basicio import parallel_print
from ..utils.physical_parameters import PhysicalParameters
from ..utils.typing import AbsorbingBCsType, AdjointType


def backward_wave_propagator(
    wave: Wave,
    dt: float = None,
    adjoint_type: AdjointType = AdjointType.IMPLEMENTED_ADJOINT,
    controls: PhysicalParameters | None = None,
) -> fire.Function | PhysicalParameters:
    """Propagates the adjoint wave backwards in time.

    Currently uses central differences.

    Parameters:
    -----------
    wave : Wave
        Wave object that already propagated a forward wave.
    dt : float (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.
    adjoint_type : AdjointType, optional
        Implemented adjoint variant to use: the hand-derived adjoint of the
        acoustic wave, or the adjoint derived by UFL differentiation of the
        forward residual form the solver exposes.
    controls : PhysicalParameters, optional
        Physical parameters the UFL-derived gradient is taken with respect
        to, as returned by ``wave.physical_parameters.select()``. Each must
        be a Firedrake ``Function`` the forward residual form depends on.
        Required by the UFL-derived adjoint and ignored by the hand-derived
        one, which differentiates with respect to the velocity model.

    Returns:
    --------
    dJ : firedrake.Function or PhysicalParameters
        Calculated gradient. The hand-derived adjoint returns the velocity
        gradient as one ``Function``; the UFL-derived adjoint returns one
        gradient ``Function`` per control, keyed by physical parameter.

    Raises:
    -------
    ValueError
        If the UFL-derived adjoint is requested without a forward residual
        form or without controls.

    Notes:
    ------
    This is an unified backward wave propagation for both PML and no-PML cases.
    The hand-derived PML path uses the mixed-space gradient form
    ``2c * ∇u_adj · ∇u_fwd`` while its no-PML path uses
    ``-2/c³ * ü_fwd * u_adj``; both advance the prebuilt forward variational
    solver with the receiver misfit as source. The UFL-derived path solves
    the discrete adjoint of the forward time step and accumulates
    ``d/dm <R(u^{n+1}, u^n, u^{n-1}; m), λ^{n+1}>`` at every step.
    """
    wave.reset_adjoint_state()
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

    use_ufl_differentiation = adjoint_type.is_ufl_derived
    if use_ufl_differentiation:
        controls = _require_ufl_differentiation_inputs(wave, controls)
        # One reduced-gradient accumulator per control, in the control's
        # space. Every backward step adds
        #
        #     dJ/dm <- dJ/dm + d/dm <R(u^{n+1}, u^n, u^{n-1}; m), λ^{n+1}>.
        dJ = PhysicalParameters(
            (parameter, fire.Function(control.function_space()))
            for parameter, control in controls.items()
        )
    else:
        dJ = fire.Function(wave.get_scalar_function_space())

    # The receiver misfit is injected as the adjoint source by the transpose
    # of the interpolation the forward solve read the receivers with: the
    # vertex-only mesh interpolation, or the Dirac delta projection.
    receiver_source_space = wave.get_adjoint_receiver_source_space()
    receivers = wave.receivers
    if wave.use_vertex_only_mesh:
        inject_receivers = receivers.receiver_source_injector(receiver_source_space)
    else:
        rhs_forcing = fire.Cofunction(receiver_source_space.dual())

        def inject_receivers(misfit_step):
            rhs_forcing.assign(0.0)
            return receivers.apply_receivers_as_source(
                rhs_forcing, wave.misfit, misfit_step,
            )

    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        wave, mask_available, controls,
    )
    if use_ufl_differentiation:
        adjoint_solver = build_adjoint_solver(
            wave.forward_residual_form,
            wave.forward_residual_states,
            wave.vstate,
            wave.prev_vstate,
            wave.next_vstate,
            wave.get_adjoint_source(),
            wave.solver_parameters,
            bcs=wave.forward_residual_bcs,
        )
    else:
        adjoint_solver = wave.solver

    forward_solution = wave.forward_solution

    for step in range(nt - 1, -1, -1):
        if wave.use_vertex_only_mesh:
            misfit_form = inject_receivers(wave.misfit[step])
        else:
            misfit_form = inject_receivers(step)
        if step == 0 or step == nt - 1:
            misfit_form.assign(0.5 * misfit_form)
        if use_ufl_differentiation:
            wave.set_adjoint_source(misfit_form)
        else:
            wave.rhs_no_pml_source().assign(misfit_form)
        adjoint_solver.solve()

        if step % wave.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(wave.get_function(state=wave.next_vstate))

            if use_ufl_differentiation:
                # The stored forward solution holds u^{k+1} at index k, so
                # the residual of this step reads u^{n+1} from the last
                # stored sample and u^n, u^{n-1} from the two before it,
                # which are zero before the first step.
                residual_np1, residual_n, residual_nm1 = (
                    wave.forward_residual_states
                )
                residual_np1.assign(forward_solution.pop())
                if len(forward_solution) > 0:
                    residual_n.assign(forward_solution[-1])
                else:
                    residual_n.assign(0.0)
                if len(forward_solution) > 1:
                    residual_nm1.assign(forward_solution[-2])
                else:
                    residual_nm1.assign(0.0)
                for control_solver in grad_solver.values():
                    control_solver.solve()
                for parameter, gradient in dJ.items():
                    gradient += gradi[parameter]
            else:
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
                    forward_field.assign(
                        _compute_dufordt2(forward_solution, sample_dt)
                    )
                grad_solver.solve()
                _trapezoidal_gradient_integration(dJ, gradi, step, last_sample)

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        t = step * float(dt)

    wave.adjoint_solution = uadj
    wave.current_time = t

    helpers.display_progress(wave.comm, t)

    if use_ufl_differentiation:
        # The discrete functional weights every step by dt.
        for gradient in dJ.values():
            gradient.assign(dt * gradient)
    else:
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


def _build_gradient_solver(
    wave: Wave,
    mask_available: bool,
    controls: PhysicalParameters | None,
) -> tuple:
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
    controls : PhysicalParameters or None
        Controls of the UFL-derived gradient, one Riesz solve each, or
        ``None`` for the hand-derived velocity gradient.

    Returns:
    --------
    grad_solver, forward_field, uadj, gradi
        For the hand-derived gradient, the Riesz solver, the forward field it
        reads, the adjoint field it reads and the per-step gradient it
        writes. For the UFL-derived gradient, ``grad_solver`` and ``gradi``
        are keyed by control and ``forward_field`` is ``None``: the forward
        state is read from the formal residual states instead.
    """
    if controls is not None:
        dx = fire.dx(**wave.quadrature_rule)
        state_space = wave.get_adjoint_receiver_source_space()
        uadj = fire.Function(state_space)
        # The residual lives on the state space, so it is paired with the
        # adjoint on that space: the full mixed adjoint with a PML, whose
        # receivers read only the pressure component, and ``uadj`` otherwise.
        if wave.abc_type == AbsorbingBCsType.PML:
            adjoint_field = wave.next_vstate
        else:
            adjoint_field = uadj

        grad_solver = {}
        gradi = {}
        for parameter, control in controls.items():
            grad_solver[parameter], gradi[parameter] = (
                _build_single_control_gradient(
                    wave, control, adjoint_field, dx,
                )
            )
        parallel_print(
            "Using UFL-derived gradient from forward residual form",
            wave.comm,
        )
        return grad_solver, None, uadj, gradi

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
            2.0 * wave.c * indicator * fire.dot(
                fire.grad(uadj), fire.grad(forward_field)) * m_v * dx
        )
        # The hand-derived PML gradient is inconsistent with the reformulated
        # PML above. AdjointType.UFL_DERIVED_ADJOINT derives the gradient
        # from the forward residual form instead.
        raise ValueError("PML gradient calculation temporarily unavailable")

    else:
        ffG = (
            -2 * (wave.c) ** (-3) * fire.dot(forward_field, uadj) * m_v * dx
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


def _build_single_control_gradient(
    wave: Wave, control: fire.Function, adjoint_field: fire.Function,
    dx: fire.Measure,
) -> tuple[fire.LinearVariationalSolver, fire.Function]:
    """Build the Riesz projection solver of one control's gradient.

    The gradient contribution of one time step comes from the discrete
    Lagrangian::

        g_m[v_m] = d/dm <R(u^{n+1}, u^n, u^{n-1}; m), λ^{n+1}> [v_m]

    where ``u`` is the forward time-stepping state. ``fire.action`` pairs the
    forward residual ``R`` with the adjoint field ``λ``; differentiating that
    scalar form with respect to the control ``m`` in the direction ``v_m``
    gives the variational gradient contribution of the current time step,
    projected onto the control space through its mass matrix (the L2 Riesz
    map).

    Parameters
    ----------
    wave : Wave
        Wave object exposing ``forward_residual_form``.
    control : firedrake.Function
        Control the gradient is taken with respect to.
    adjoint_field : firedrake.Function
        Adjoint state paired with the forward residual.
    dx : ufl.Measure
        Volume measure carrying the solver's quadrature rule.

    Returns
    -------
    tuple
        ``(grad_solver, gradi)`` where ``gradi`` receives the per-step gradient.
    """
    control_space = control.function_space()
    trial = fire.TrialFunction(control_space)
    test = fire.TestFunction(control_space)
    mass = trial * test * dx
    dRdm = fire.derivative(
        fire.action(wave.forward_residual_form, adjoint_field),
        control,
        test,
    )
    gradi = fire.Function(control_space)
    grad_problem = fire.LinearVariationalProblem(mass, dRdm, gradi)
    grad_solver = fire.LinearVariationalSolver(
        grad_problem,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )
    return grad_solver, gradi


def build_adjoint_solver(
    forward_residual_form,
    forward_residual_states: tuple,
    adjoint_current_state: fire.Function,
    adjoint_previous_state: fire.Function,
    adjoint_next_state: fire.Function,
    adjoint_source: fire.Cofunction,
    solver_parameters: dict,
    bcs=(),
) -> fire.LinearVariationalSolver:
    """Build a one-step adjoint solver from a discrete forward residual.

    The input residual represents one forward time step,

        R(u^{n+1}, u^n, u^{n-1}; m) = 0.

    UFL differentiation gives the linearized blocks

        R_{u^{n+1}}, R_{u^n}, R_{u^{n-1}},

    and the discrete adjoint step solves

        R_{u^{n+1}}^T λ^{n+1}
            = -R_{u^n}^T λ^n
              -R_{u^{n-1}}^T λ^{n-1}
              + J_u.

    Parameters
    ----------
    forward_residual_form : ufl.Form
        Forward residual form for one time step.
    forward_residual_states : tuple
        Formal residual states corresponding to ``u^{n+1}``, ``u^n`` and
        ``u^{n-1}``.
    adjoint_current_state : firedrake.Function
        Current adjoint state, ``λ^n``.
    adjoint_previous_state : firedrake.Function
        Previous adjoint state, ``λ^{n-1}``.
    adjoint_next_state : firedrake.Function
        Unknown adjoint state solved by this step, ``λ^{n+1}``.
    adjoint_source : firedrake.Cofunction
        Source term representing the derivative of the objective with respect
        to the state.
    solver_parameters : dict
        Firedrake/PETSc solver parameters.
    bcs : iterable of firedrake.DirichletBC, optional
        Dirichlet boundary conditions of the forward solve. The adjoint
        state satisfies their homogeneous counterparts.

    Returns
    -------
    firedrake.LinearVariationalSolver
        Solver advancing the adjoint state by one step.
    """
    residual_np1, residual_n, residual_nm1 = forward_residual_states
    state_space = residual_np1.function_space()
    direction = fire.TrialFunction(state_space)

    dR_dnp1 = fire.derivative(
        forward_residual_form, residual_np1, direction,
    )
    dR_dn = fire.derivative(
        forward_residual_form, residual_n, direction,
    )
    dR_dnm1 = fire.derivative(
        forward_residual_form, residual_nm1, direction,
    )

    adjoint_lhs = fire.adjoint(dR_dnp1)
    adjoint_rhs = (
        -fire.action(fire.adjoint(dR_dn), adjoint_current_state)
        - fire.action(fire.adjoint(dR_dnm1), adjoint_previous_state)
    )
    problem = fire.LinearVariationalProblem(
        adjoint_lhs,
        adjoint_rhs + adjoint_source,
        adjoint_next_state,
        bcs=[fire.homogenize(bc) for bc in bcs],
        constant_jacobian=True,
    )
    solver_parameters = dict(solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    return fire.LinearVariationalSolver(
        problem,
        solver_parameters=solver_parameters,
    )


def _require_ufl_differentiation_inputs(
    wave: Wave, controls: PhysicalParameters | None,
) -> PhysicalParameters:
    """Return the UFL-derived controls, or raise for missing inputs.

    Parameters
    ----------
    wave : Wave
        Wave object expected to expose a forward residual form.
    controls : PhysicalParameters or None
        Controls selected for the gradient.

    Returns
    -------
    PhysicalParameters
        The controls, every one a Firedrake ``Function``.

    Raises
    ------
    ValueError
        If the wave exposes no forward residual form or states, or if no
        control or a control that is not a ``Function`` is given.
    """
    if wave.forward_residual_form is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires "
            "wave.forward_residual_form."
        )
    if wave.forward_residual_states is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires "
            "wave.forward_residual_states."
        )
    if not controls:
        raise ValueError(
            "UFL-derived implemented adjoint requires at least one physical "
            "parameter as control, selected with "
            "wave.physical_parameters.select()."
        )
    for parameter, control in controls.items():
        if not isinstance(control, fire.Function):
            raise ValueError(
                f"Control '{parameter.value}' must be a Firedrake Function "
                f"the forward residual form depends on, received "
                f"{type(control).__name__}."
            )
    return controls


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
