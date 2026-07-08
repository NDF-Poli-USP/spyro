import firedrake as fire
from . import helpers
from .wave import Wave
from ..io.basicio import parallel_print
from ..utils.typing import AdjointType, ImplementedAdjointDerivation

# Key used to store a single-control (e.g. acoustic velocity) gradient in the
# same control-keyed dictionary used for multi-parameter (elastic) controls.
# It lets the whole UFL-derived backward pass work with one dict layout and
# unwrap to a single Function only at the public API boundary.
_SINGLE_CONTROL_KEY = "control"


def backward_wave_propagator(
    wave_obj: Wave,
    dt: float = None,
    adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
):
    """Propagates the adjoint wave backwards in time.

    Currently uses central differences.

    Parameters:
    -----------
    wave_obj : Wave
        Wave object that already propagated a forward wave.
    dt : float (optional)
        Time step to be used explicitly. If not mentioned uses the default,
        that was estabilished in the wave object for the adjoint model.
    adjoint_type : AdjointType, optional
        Implemented adjoint variant to use.

    Returns:
    --------
    dJ : firedrake.Function or dict
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
    wave_obj.reset_adjoint_state()
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

    use_ufl_differentiation = (
        adjoint_type.implemented_derivation
        is ImplementedAdjointDerivation.UFL_DIFFERENTIATION
    )
    form_controls = None
    if use_ufl_differentiation:
        form_controls = _require_ufl_differentiation_inputs(wave_obj)
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
    if use_ufl_differentiation:
        dJ = {
            parameter: fire.Function(control.function_space())
            for parameter, control in _control_map(form_controls).items()
        }
    else:
        dJ = fire.Function(wave_obj.get_scalar_function_space())
    # The UFL-derived path works internally with a control-keyed dict, even for
    # single-control solvers. ``controls_are_dict`` records whether the public
    # API should return that dict (elastic) or a single Function (acoustic).
    controls_are_dict = use_ufl_differentiation and isinstance(form_controls, dict)
    receiver_source_space = wave_obj.get_adjoint_receiver_source_space()
    rhs_forcing = None
    if not use_ufl_differentiation:
        rhs_forcing = fire.Cofunction(receiver_source_space.dual())
    grad_solver, forward_field, uadj, gradi = _build_gradient_solver(
        wave_obj, mask_available, use_ufl_differentiation, form_controls,
    )
    if use_ufl_differentiation:
        adjoint_solver = build_adjoint_solver(
            wave_obj.forward_residual_form,
            wave_obj.forward_residual_states,
            wave_obj.vstate,
            wave_obj.prev_vstate,
            wave_obj.next_vstate,
            wave_obj.get_adjoint_source(),
            wave_obj.solver_parameters,
        )
    else:
        adjoint_solver = wave_obj.solver

    forward_solution = wave_obj.forward_solution
    receivers = wave_obj.receivers

    for step in range(nt - 1, -1, -1):
        if use_ufl_differentiation:
            misfit_form = receivers.apply_receivers_as_source_vertex_only_mesh(
                wave_obj.misfit[step], receiver_source_space,
            )
        else:
            rhs_forcing.assign(0.0)
            misfit_form = receivers.apply_receivers_as_source(
                rhs_forcing, wave_obj.misfit, step,
            )
        if step == 0 or step == nt - 1:
            misfit_form.assign(0.5 * misfit_form)
        wave_obj.set_adjoint_source(misfit_form)
        adjoint_solver.solve()

        if step % wave_obj.gradient_sampling_frequency == 0:
            # Assign the adjoint solution at the step `np1` to `uadj`.
            uadj.assign(wave_obj.get_function(state=wave_obj.next_vstate))

            if use_ufl_differentiation:
                residual_np1, residual_n, residual_nm1 = (
                    wave_obj.forward_residual_states
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
            elif wave_obj.abc_boundary_layer_type == "PML":
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
            if use_ufl_differentiation:
                for control_solver in grad_solver.values():
                    control_solver.solve()
                for parameter in dJ:
                    dJ[parameter] += gradi[parameter]
            else:
                grad_solver.solve()
                _trapezoidal_gradient_integration(dJ, gradi, step, nt)

        wave_obj.prev_vstate = wave_obj.vstate
        wave_obj.vstate = wave_obj.next_vstate
        t = step * float(dt)

    wave_obj.adjoint_solution = uadj
    wave_obj.current_time = t

    helpers.display_progress(wave_obj.comm, t)

    if use_ufl_differentiation:
        for gradient in dJ.values():
            gradient.assign(dt * gradient)
        if not controls_are_dict:
            dJ = dJ[_SINGLE_CONTROL_KEY]
    else:
        dJ.assign((dt / 2) * dJ)
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


def _build_gradient_solver(
    wave_obj: Wave,
    mask_available: bool,
    use_ufl_differentiation: bool,
    form_controls,
) -> tuple[
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
    use_ufl_differentiation : bool
        Whether to derive the adjoint and gradient from the forward residual.
    form_controls : firedrake.Function or dict or None
        Control parameter(s) used by the UFL residual backend.

    Returns:
    --------
    grad_solver, forward_field, uadj, gradi
    """
    if use_ufl_differentiation:
        dx = fire.dx(**wave_obj.quadrature_rule)
        state_space = wave_obj.get_adjoint_receiver_source_space()
        uadj = fire.Function(state_space)
        # For PML the residual lives on the mixed state space, so it must be
        # paired with the full mixed adjoint. Without PML the residual is scalar
        # and pairs with the scalar adjoint field.
        if wave_obj.abc_boundary_layer_type == "PML":
            adjoint_field = wave_obj.next_vstate
        else:
            adjoint_field = uadj

        grad_solver = {}
        gradi = {}
        for parameter, control in _control_map(form_controls).items():
            grad_solver[parameter], gradi[parameter] = (
                _build_single_control_gradient(
                    wave_obj, control, adjoint_field, dx,
                )
            )
        parallel_print(
            "Using UFL-derived gradient from forward residual form",
            wave_obj.comm,
        )
        return grad_solver, None, uadj, gradi

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

    uadj = fire.Function(V)
    forward_field = fire.Function(V)
    mgrad = m_u * m_v * dx

    if wave_obj.abc_boundary_layer_type == "PML":
        # Always exclude PML region from gradient.
        # This is necessary once the gradient expression is not considering
        # the PML auxiliary variables. In addition, we are not interested
        # in the gradient in the PML region.
        indicator = _pml_interior_indicator(wave_obj)
        # Compute the gradient only in the physical domain.
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


def _build_single_control_gradient(wave_obj, control, adjoint_field, dx):
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
    wave_obj : Wave
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
        fire.action(wave_obj.forward_residual_form, adjoint_field),
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
        constant_jacobian=True,
    )
    solver_parameters = dict(solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    return fire.LinearVariationalSolver(
        problem,
        solver_parameters=solver_parameters,
    )


def _get_form_controls(wave_obj: Wave):
    """Return controls supported by the UFL residual backend.

    ``Wave.get_control_parameters()`` is the solver-level API for inversion
    controls. Acoustic waves expose one control, the velocity model, as a
    Firedrake ``Function``. Isotropic elastic waves expose a dictionary of
    scalar material controls, so each entry gets its own Riesz solve.
    """
    try:
        controls = wave_obj.get_control_parameters()
    except NotImplementedError:
        return None

    if isinstance(controls, fire.Function):
        return controls
    if isinstance(controls, dict) and all(
        isinstance(control, fire.Function) for control in controls.values()
    ):
        return controls
    return None


def _require_ufl_differentiation_inputs(wave_obj: Wave):
    """Return UFL controls or raise for missing UFL-derived adjoint inputs."""
    if wave_obj.forward_residual_form is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires "
            "wave.forward_residual_form."
        )
    if wave_obj.forward_residual_states is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires "
            "wave.forward_residual_states."
        )

    controls = _get_form_controls(wave_obj)
    if controls is None:
        raise ValueError(
            "UFL-derived implemented adjoint requires Firedrake Function "
            "control parameters from wave.get_control_parameters()."
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
