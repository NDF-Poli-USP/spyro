from collections.abc import Callable

import firedrake as fire
from spyro.mpi.spyro_mpi import SpyroEnsemble
from . import helpers
from .wave import Wave
from ..utils.physical_parameters import PhysicalParameters
from ..utils.typing import (
    AbsorbingBCsType, AcousticMaterialParameter, AdjointType,
    ImplementedAdjointDerivation,
)

# Key used to store a single-control (e.g. acoustic velocity) gradient in the
# same control-keyed dictionary used for multi-parameter (elastic) controls.
# It lets the whole UFL-derived backward pass work with one dict layout and
# unwrap to a single Function only at the public API boundary.
_SINGLE_CONTROL_KEY = "control"


def backward_wave_propagator(
    wave: Wave,
    dt: float = None,
    adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    step_callback: Callable[[int, fire.Function], None] | None = None,
    compute_gradient: bool = True,
):
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
        Implemented adjoint variant to use.
    step_callback : callable (optional)
        Called as ``step_callback(step, adjoint)`` right after the adjoint of
        each time step is solved, in reverse order of the steps. ``adjoint``
        is the adjoint state of the forward solve of that step; it is
        overwritten afterwards, so copy it to keep it.
    compute_gradient : bool (optional)
        Accumulate the gradient. Default True. With False only the adjoint
        is propagated (for ``step_callback``), the stored forward solution is
        not needed and nothing is returned.

    Returns:
    --------
    dJ : firedrake.Function or PhysicalParameters or None
        Calculated gradient. Acoustic controls return one ``Function``.
        Isotropic elastic controls return a dictionary of ``Function`` objects
        keyed by material parameter.

    Notes:
    ------
    This is an unified backward wave propagation for both PML and no-PML cases.
    The implemented adjoint can use either UFL differentiation of the forward
    residual or the legacy hand-derived adjoint/gradient forms, selected by
    ``adjoint_type``.
    """
    wave.reset_adjoint_state()
    mask_available = wave.gradient_mask_available
    if dt is not None:
        wave.dt = dt

    dt = wave.dt
    t = wave.current_time
    if t != wave.final_time:
        SpyroEnsemble.print(
            f"Current time of {t}, different than final_time of "
            f"{wave.final_time}. Setting final_time to current time "
            f"in backwards propagation."
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

    use_ufl_differentiation = (
        adjoint_type.implemented_derivation
        is ImplementedAdjointDerivation.UFL_DIFFERENTIATION
    )
    form_controls = None
    if use_ufl_differentiation:
        if compute_gradient and freq != 1:
            # The residual of step n pairs u^{n+1}, u^n and u^{n-1}, so every
            # forward step must be stored.
            raise ValueError(
                "The UFL-derived implemented adjoint needs "
                "gradient_sampling_frequency = 1."
            )
        form_controls = _require_ufl_differentiation_inputs(
            wave, need_controls=compute_gradient,
        )
    # Initialize the reduced gradient accumulator with zero in the control
    # space(s).  During the backward time loop each step adds a contribution
    #
    #     dJ/dm <- dJ/dm + dJ_step/dm,
    #
    # where, for the UFL-derived path,
    #
    #     dJ_step/dm [dm] =
    #         d/dm <R(y^{n+1}, y^n, y^{n-1}; m), lambda^{n+1}> [dm].
    #
    # Acoustic has a single control and is wrapped internally under
    # ``_SINGLE_CONTROL_KEY``; elastic exposes one accumulator per material
    # parameter. The public return value is unwrapped back to a single Function
    # for acoustic at the end of the routine.
    if not compute_gradient:
        dJ = None
    elif use_ufl_differentiation:
        dJ = {
            parameter: fire.Function(control.function_space())
            for parameter, control in _control_map(form_controls).items()
        }
    else:
        dJ = fire.Function(wave.get_scalar_function_space())
    # The UFL-derived path works internally with a control-keyed dict, even for
    # single-control solvers. ``controls_are_dict`` records whether the public
    # API should return that dict (elastic) or a single Function (acoustic).
    controls_are_dict = use_ufl_differentiation and isinstance(form_controls, dict)
    receiver_source_space = wave.get_adjoint_receiver_source_space()
    rhs_forcing = None
    if not use_ufl_differentiation and not wave.use_vertex_only_mesh:
        rhs_forcing = fire.Cofunction(receiver_source_space.dual())
    if compute_gradient:
        grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
            wave, mask_available, use_ufl_differentiation, form_controls,
        )
    else:
        uadj = fire.Function(receiver_source_space)
    if use_ufl_differentiation:
        adjoint_solver = build_adjoint_solver(
            wave.forward_residual_form,
            wave.forward_residual_states,
            wave.vstate,
            wave.prev_vstate,
            wave.next_vstate,
            wave.get_adjoint_source(),
            wave.solver_parameters,
            bcs=getattr(wave, "bcs", None),
        )
    else:
        adjoint_solver = wave.solver

    forward_solution = wave.forward_solution
    receivers = wave.receivers

    for step in range(nt - 1, -1, -1):
        if use_ufl_differentiation or wave.use_vertex_only_mesh:
            misfit_form = receivers.apply_receivers_as_source_vertex_only_mesh(
                wave.misfit[step], receiver_source_space,
            )
        else:
            rhs_forcing.assign(0.0)
            misfit_form = receivers.apply_receivers_as_source(
                rhs_forcing, wave.misfit, step,
            )
        if step == 0 or step == nt - 1:
            misfit_form.assign(0.5 * misfit_form)
        if use_ufl_differentiation:
            wave.set_adjoint_source(misfit_form)
        else:
            wave.rhs_no_pml_source().assign(misfit_form)
        adjoint_solver.solve()
        if step_callback is not None:
            step_callback(step, wave.get_function(state=wave.next_vstate))

        if compute_gradient and step % wave.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(wave.get_function(state=wave.next_vstate))

            if use_ufl_differentiation:
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
            elif wave.abc_type == AbsorbingBCsType.PML:
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
            if use_ufl_differentiation:
                for control_solver in grad_solver.values():
                    control_solver.solve()
                for parameter in dJ:
                    dJ[parameter] += gradi[parameter]
            else:
                grad_solver.solve()
                _trapezoidal_gradient_integration(dJ, gradi, step, last_sample)

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        t = step * float(dt)

    if not compute_gradient:
        uadj.assign(wave.get_function(state=wave.vstate))
    wave.adjoint_solution = uadj
    wave.current_time = t

    helpers.display_progress(wave.comm, t)

    if not compute_gradient:
        return None

    if use_ufl_differentiation:
        for gradient in dJ.values():
            gradient.assign(dt * gradient)
        if controls_are_dict:
            dJ = PhysicalParameters(dJ)
        else:
            dJ = dJ[_SINGLE_CONTROL_KEY]
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
    use_ufl_differentiation: bool,
    form_controls,
) -> tuple[
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
    use_ufl_differentiation : bool
        Whether to derive the adjoint and gradient from the forward residual.
    form_controls : firedrake.Function or dict or None
        Control parameter(s) used by the UFL residual backend.

    Returns:
    --------
    grad_solver, forward_field, uadj, gradi
    """
    if use_ufl_differentiation:
        dx = fire.dx(**wave.quadrature_rule)
        state_space = wave.get_adjoint_receiver_source_space()
        uadj = fire.Function(state_space)
        # For PML the residual lives on the mixed state space, so it must be
        # paired with the full mixed adjoint. Without PML the residual is scalar
        # and pairs with the scalar adjoint field.
        if wave.abc_type == AbsorbingBCsType.PML:
            adjoint_field = wave.next_vstate
        else:
            adjoint_field = uadj

        grad_solver = {}
        gradi = {}
        for parameter, control in _control_map(form_controls).items():
            grad_solver[parameter], gradi[parameter] = (
                _build_single_control_gradient(
                    wave, control, adjoint_field, dx,
                )
            )
        SpyroEnsemble.print(
            "Using UFL-derived gradient from forward residual form"
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

    uadj = fire.Function(V)
    forward_field = fire.Function(V)
    mgrad = m_u * m_v * dx

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
        # PML above. Use AdjointType with UFL_DIFFERENTIATION instead, which
        # derives the gradient from the forward residual form.
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


def _build_single_control_gradient(wave, control, adjoint_field, dx):
    """Build the Riesz projection solver for one control's gradient.

    The gradient contribution comes from the discrete Lagrangian::

        g_m[v_m] = d/dm <R(y^{n+1}, y^n, y^{n-1}; m), lambda^{n+1}> [v_m]

    where ``y`` is the forward time-stepping state. ``fire.action`` pairs the
    forward residual ``R`` with the adjoint field ``lambda``; differentiating
    that scalar form with respect to the control ``m`` in direction ``v_m``
    gives the variational gradient contribution for the current time step. The
    contribution is projected onto the control space through the mass matrix
    (an L2 Riesz map).

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
    forward_residual_states,
    adjoint_current_state,
    adjoint_previous_state,
    adjoint_next_state,
    adjoint_source,
    solver_parameters,
    bcs=None,
) -> fire.LinearVariationalSolver:
    """Build a one-step adjoint solver from a discrete forward residual.

    The input residual represents one forward time step,

        R(y^{n+1}, y^n, y^{n-1}; m) = 0.

    UFL differentiation gives the linearized blocks

        R_{y^{n+1}}, R_{y^n}, R_{y^{n-1}},

    and the discrete adjoint step solves

        R_{y^{n+1}}^T lambda^{n+1}
            = -R_{y^n}^T lambda^n
              -R_{y^{n-1}}^T lambda^{n-1}
              + J_y.

    Parameters
    ----------
    forward_residual_form : ufl.Form
        Forward residual form for one time step.
    forward_residual_states : tuple
        Formal residual states corresponding to ``y^{n+1}``, ``y^n`` and
        ``y^{n-1}``.
    adjoint_current_state : firedrake.Function
        Current adjoint state, ``lambda^n``.
    adjoint_previous_state : firedrake.Function
        Previous adjoint state, ``lambda^{n-1}``.
    adjoint_next_state : firedrake.Function
        Unknown adjoint state solved by this step, ``lambda^{n+1}``.
    adjoint_source : firedrake.Cofunction
        Source term representing the derivative of the objective with respect
        to the state.
    solver_parameters : dict
        Firedrake/PETSc solver parameters.
    bcs : list of firedrake.DirichletBC, optional
        Dirichlet conditions of the forward step. The adjoint takes their
        homogeneous version on the same boundaries: a prescribed value is
        not a degree of freedom, so its adjoint vanishes there.

    Returns
    -------
    firedrake.LinearVariationalSolver
        Solver of one adjoint step.
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
    adjoint_bcs = [
        fire.DirichletBC(bc.function_space(), 0.0, bc.sub_domain)
        for bc in (bcs or [])
    ]
    problem = fire.LinearVariationalProblem(
        adjoint_lhs,
        adjoint_rhs + adjoint_source,
        adjoint_next_state,
        bcs=adjoint_bcs or None,
        constant_jacobian=True,
    )
    solver_parameters = dict(solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    return fire.LinearVariationalSolver(
        problem,
        solver_parameters=solver_parameters,
    )


def _get_form_controls(wave: Wave):
    """Return the controls of the UFL-derived gradient.

    The controls are the independent physical parameters selected with
    :meth:`Wave.enable_implemented_adjoint` (all of them by default). An
    acoustic wave has one control, the velocity, returned as a Firedrake
    ``Function``; any other selection is returned as a dictionary keyed by
    material parameter, so each entry gets its own Riesz solve.

    Parameters
    ----------
    wave : Wave
        Wave whose physical parameters are initialized.

    Returns
    -------
    firedrake.Function or dict or None
        The controls, or ``None`` when the wave has no physical parameters.
    """
    try:
        parameters = wave.physical_parameters
    except ValueError:
        return None
    controls = parameters.select(wave.implemented_adjoint_control_parameters)
    names = list(controls)
    if len(names) == 1 and isinstance(names[0], AcousticMaterialParameter):
        return controls[names[0]]
    return dict(controls.items())


def _require_ufl_differentiation_inputs(wave: Wave, need_controls: bool = True):
    """Return UFL controls or raise for missing UFL-derived adjoint inputs.

    Parameters
    ----------
    wave : Wave
        Wave whose forward residual drives the adjoint.
    need_controls : bool, optional
        Whether the controls are needed (a gradient is computed). Default
        ``True``; with ``False`` only the residual is checked.

    Returns
    -------
    firedrake.Function or dict or None
        The controls, or ``None`` when they are not needed.
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

    if not need_controls:
        return None
    controls = _get_form_controls(wave)
    if controls is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires initialized physical "
            "parameters (wave.physical_parameters)."
        )
    return controls


def _control_map(controls) -> dict:
    """Return UFL-derived controls as a control-keyed dictionary.

    Multi-parameter solvers (isotropic elastic) already expose a dictionary.
    Single-control solvers (acoustic) are wrapped under ``_SINGLE_CONTROL_KEY``
    so the rest of the backward pass can treat every case uniformly.
    """
    if isinstance(controls, dict):
        return controls
    return {_SINGLE_CONTROL_KEY: controls}


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
