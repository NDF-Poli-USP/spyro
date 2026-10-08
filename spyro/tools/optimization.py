"""Tools to minimize a reduced functional with pyadjoint's ``TAOSolver``: the
metric of the controls, the bounds in the form TAO expects, and the call to
TAO itself.
"""

import warnings

import firedrake as fire
import numpy as np

from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface,
    TAOConvergenceError,
)
from pyadjoint.reduced_functional import AbstractReducedFunctional

from ..functionals.reduced import LatentReducedFunctional, LumpedL2ReducedFunctional
from ..functionals.reduced.lumped_l2 import _inverse_sqrt_lumped_mass
from ..utils.physical_parameters import as_list


def tao_bounds(bound, controls):
    """Shape a bound specification into what TAO takes.

    TAO takes one bound per control, each a scalar it broadcasts over that
    control or a field in the control's own space. A scalar bounds every
    control the same way; a sequence gives one entry per control, which is
    what an optimization over parameters of different scales needs. For a
    single control, a sequence longer than one entry is read as varying per
    degree of freedom instead.

    Parameters
    ----------
    bound : scalar or array_like
        Bound specification.
    controls : firedrake.Function or list of firedrake.Function
        The controls being bounded, in the order TAO takes them.

    Returns
    -------
    list
        One bound for each control, each a ``float`` or a ``Function``.

    Raises
    ------
    ValueError
        If a sequence of bounds does not have one entry per control, or an
        entry does not match the size of the control it bounds.
    """
    controls = as_list(controls)
    if np.isscalar(bound):
        return [float(bound)] * len(controls)

    bounds = list(bound)
    if len(controls) == 1 and len(bounds) != 1:
        # A lone control takes its bounds one per degree of freedom.
        bounds = [bound]
    if len(bounds) != len(controls):
        raise ValueError(
            f"{len(controls)} controls are being optimized, so the bounds "
            f"take that many entries; {len(bounds)} were given.",
        )

    shaped = []
    for value, control in zip(bounds, controls):
        if np.isscalar(value):
            shaped.append(float(value))
            continue
        # A bound that varies over the mesh becomes a field of its own, in
        # the space of the control it bounds.
        shape = np.asarray(control.dat.data_ro).shape
        size = int(np.prod(shape))
        data = np.asarray(value, dtype=float).reshape(-1)
        if data.size == 1:
            data = np.full(size, data[0])
        if data.size != size:
            raise ValueError(
                f"A bound on '{control.name()}' has {data.size} entries, "
                f"and the control has {size}.",
            )
        shaped.append(fire.Function(
            control.function_space(), name=control.name(),
            val=data.reshape(shape),
        ))
    return shaped


def _bound_to_field(bound, inverse_sqrt_mass: fire.Function) -> fire.Function | None:
    r"""Return a bound on the model as a field on the controls TAO optimizes.

    The bound on :math:`m` becomes a field on :math:`\tilde{m} = M_L^{1/2} m`.

    Parameters
    ----------
    bound : float, firedrake.Function or None
        The bound on :math:`m`.
    inverse_sqrt_mass : firedrake.Function
        :math:`M_L^{-1/2}` in the space of the control it bounds.

    Returns
    -------
    firedrake.Function or None
        The bound on :math:`\tilde{m}`, or None for no bound.
    """
    if bound is None:
        return None
    field = fire.Function(inverse_sqrt_mass.function_space())
    values = bound.dat.data_ro if isinstance(bound, fire.Function) else bound
    field.dat.data_wo[:] = values / inverse_sqrt_mass.dat.data_ro
    return field


def _lumped_bounds(bounds: list, controls) -> list:
    r"""Return the bounds on the model as bounds on the controls TAO optimizes.

    Bounds on :math:`m` become bounds on :math:`\tilde{m} = M_L^{1/2} m`.
    The change of variables is diagonal and positive, so a box on :math:`m`
    is a box on :math:`\tilde{m}`, with each bound scaled the same way.

    Parameters
    ----------
    bounds : sequence of tuple
        ``(lower, upper)`` on :math:`m`, one per control. Each bound is a
        scalar, a field, or None.
    controls : pyadjoint.enlisting.Enlist
        Controls of the :class:`LumpedL2ReducedFunctional`, over
        :math:`\tilde{m}`.

    Returns
    -------
    list of tuple
        ``(lower, upper)`` on :math:`\tilde{m}`, one per control.
    """
    # One M_L^{-1/2} per space: controls in the same space share it.
    inverse_sqrt_masses = {}
    m_tilde_bounds = []
    for (lower, upper), control in zip(bounds, controls):
        space = control.control.function_space()
        if space not in inverse_sqrt_masses:
            inverse_sqrt_masses[space] = _inverse_sqrt_lumped_mass(space)
        inverse_sqrt_mass = inverse_sqrt_masses[space]
        m_tilde_bounds.append((
            _bound_to_field(lower, inverse_sqrt_mass),
            _bound_to_field(upper, inverse_sqrt_mass),
        ))
    return m_tilde_bounds


def minimize_with_tao(
    reduced_functional: AbstractReducedFunctional,
    bounds: list | None = None,
    comm=None,
    options: dict | None = None,
    record=None,
    latent: bool = False,
) -> list:
    """Minimize a reduced functional with PETSc TAO, in the lumped L2 metric.

    The optimization runs over :class:`LumpedL2ReducedFunctional`, so TAO
    measures gradients, takes steps and projects onto the bounds in the
    lumped :math:`L^2` metric of the controls. With ``latent``, the controls
    are first replaced by the latent controls of
    :class:`LatentReducedFunctional`, which keep the model inside ``bounds``
    on their own, so TAO runs without bounds in the lumped :math:`L^2`
    metric of the latent controls.

    Under ensemble parallelism the controls are replicated on every member,
    so ``comm`` has to be the *spatial* communicator: TAO's default
    (``COMM_WORLD``) would count each member's copy as separate degrees of
    freedom.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional to minimize, and the controls to minimize it over.
    bounds : list of tuple, optional
        One ``(lower, upper)`` pair per control, each a scalar TAO broadcasts
        over the control or a value in the control's own space. Required
        with ``latent``.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over.
    options : dict, optional
        PETSc options for TAO, such as ``{"tao_max_it": 20}``. The method is
        always BQNLS, TAO's bound-constrained quasi-Newton method.
    record : callable, optional
        Called ``record(iteration, functional, controls)`` after each
        iteration TAO accepts, with the controls it stands at as a list of
        fresh fields. The starting point is not reported: it is the value the
        caller already has, from evaluating the functional to get here.
        With ``latent``, the controls are the model, not the latent ones.
    latent : bool, optional
        Whether to optimize over the latent controls. Default False.

    Returns
    -------
    list
        The controls TAO stopped at, one per control of the reduced
        functional, always as a list however many there are. The solvers
        themselves return a bare control when there is only one; normalizing
        here means a caller never has to ask which case it is in.

    Warns
    -----
    UserWarning
        If TAO reaches its iteration limit. The last iterate is returned,
        since a fixed iteration budget is a normal way to run an optimization.

    Raises
    ------
    ValueError
        If the options or the PETSc command line ask for a TAO type other
        than BQNLS, or ``latent`` is given bounds that are missing,
        nonfinite or reversed.
    TAOConvergenceError
        If TAO fails for any reason other than its iteration limit, including
        line-search failure or a nonfinite objective or gradient.

    See Also
    --------
    LumpedL2ReducedFunctional : The change of variables TAO runs in.
    LatentReducedFunctional : The latent controls, with ``latent``.
    tao_bounds : Shapes ``vmin``/``vmax`` into the ``bounds`` this takes.
    """
    options = dict(options or {})
    tao_type = options.setdefault("tao_type", "bqnls")
    if tao_type != "bqnls":
        raise ValueError(
            f"minimize_with_tao always uses BQNLS, not '{tao_type}'.",
        )
    to_model = list
    if latent:
        reduced_functional = LatentReducedFunctional(reduced_functional, bounds)
        to_model = reduced_functional.map_result
        bounds = None
    lumped_functional = LumpedL2ReducedFunctional(reduced_functional)
    if bounds is not None:
        bounds = _lumped_bounds(bounds, lumped_functional.controls)
    problem = MinimizationProblem(lumped_functional, bounds=bounds)
    solver = TAOSolver(problem, options, comm=comm)
    # The PETSc command line can still set the type.
    if solver.tao.getType() != "bqnls":
        raise ValueError(
            "minimize_with_tao always uses BQNLS, not "
            f"'{solver.tao.getType()}'.",
        )
    # TAO holds an iterate as one vector with every control concatenated into
    # it. Reading it through an interface built from those same controls lays
    # them out the way the solver's own does. Both the monitor and the
    # unconverged exit below need that, so it is built once.
    controls = [control.control for control in lumped_functional.controls]
    vec_interface = PETScVecInterface(tuple(controls), comm=comm)

    def iterate_of(tao):
        """Return the model controls TAO currently stands at, as new fields."""
        iterate = [control.copy(deepcopy=True) for control in controls]
        vec_interface.from_petsc(tao.getSolution(), iterate)
        return to_model(lumped_functional.map_result(iterate))

    if record is not None:
        def monitor(tao):
            iteration, functional = tao.getSolutionStatus()[:2]
            if iteration:
                record(iteration, functional, iterate_of(tao))

        solver.tao.setMonitor(monitor)

    try:
        return to_model(lumped_functional.map_result(as_list(solver.solve())))
    except TAOConvergenceError as error:
        if solver.tao.getConvergedReason() != fire.PETSc.TAO.Reason.DIVERGED_MAXITS:
            raise
        warnings.warn(
            f"{error} Returning the last iterate; raise the iteration limit "
            "or loosen the tolerances in the TAO options if the optimization "
            "is meant to run to convergence.",
        )
        return iterate_of(solver.tao)
