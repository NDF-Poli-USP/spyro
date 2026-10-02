r"""Optimizers a reduced functional can be handed to.

What lives here is the optimizer's half of an inversion: the metric the
controls are measured in, shaping bounds into the form TAO takes them, driving
TAO, and reading the iterate back when it stops short. None of it knows what is
being inverted for. The optimizer itself is pyadjoint's ``TAOSolver``.

The metric
----------
A gradient computed by the adjoint is a *dual* object: it lives in
:math:`V'`, and turning it into a direction in :math:`V` takes the Riesz map,
:math:`\nabla J = M^{-1} DJ` with :math:`M` the mass matrix. TAO is told which
metric to measure gradients in through ``setGradientNorm``, and that is the
metric its convergence test uses too: ``tao_gatol`` and ``tao_grtol`` are read
on the projected gradient in this norm, not on the coefficients. It is what
makes a tolerance mean the same thing on a finer mesh.

For a *bound-constrained* problem that choice is not free. TAO projects onto
the box coefficient by coefficient, and a projection is only a projection in a
metric that is itself coefficient-wise. A consistent mass matrix couples
neighbouring degrees of freedom, so the projected point is not the closest
feasible point in that metric, and the two fight each other. Lumping the mass
-- collapsing it to its row sums, which is a diagonal matrix -- makes the
metric coefficient-wise too, and the two agree again. That is what
:class:`LumpedL2RieszMap` is for.

pyadjoint reads the metric off each control, ``Control(m, riesz_map=...)``:
``TAOSolver`` measures gradients with that Riesz map, and seeds the
quasi-Newton initial Hessian with it, so the first step follows the gradient
in the same metric as the convergence test. Building the controls with a
:class:`LumpedL2RieszMap` is therefore all a bound-constrained inversion needs
from here; the solver is pyadjoint's own.

This module is built on pyadjoint's TAO support, some of it internal, so
``spyro.solvers.inversion`` imports it where it drives an optimization rather
than at the top of the file. An inversion that never asks for the automated
adjoint therefore never loads this, and never depends on what it needs.
"""

import warnings

import firedrake as fire
import numpy as np
import ufl

from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface,
    TAOConvergenceError,
)

from ..domains.quadrature import quadrature_rules
from ..utils.physical_parameters import as_list


def _lumped_mass(function_space):
    r"""Return the lumped mass of a control space.

    The mass is lumped by row sums, :math:`m_i = \sum_j M_{ij}`, which is what
    ``action(u v dx, 1)`` assembles: applying the mass matrix to the constant
    one. The row sums of a mass matrix partition the domain measure, so the
    entries sum to the volume of the mesh, and each one is the measure the
    degree of freedom owns.

    Spyro's spectral elements are integrated with a quadrature of their own,
    under which the mass matrix is already diagonal and lumping is exact. That
    rule is used when the space has one; a space the rule does not cover falls
    back to the default measure, where lumping is an approximation of the
    consistent mass rather than a rewriting of it.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space a control lives in.

    Returns
    -------
    firedrake.Cofunction
        The row sums.
    """
    trial = fire.TrialFunction(function_space)
    test = fire.TestFunction(function_space)
    one = fire.Function(function_space).assign(1.0)

    try:
        quadrature, _, _ = quadrature_rules(function_space)
    except ValueError:
        quadrature = {}

    measure = fire.dx(**quadrature) if quadrature else fire.dx
    return fire.assemble(fire.action(trial * test * measure, one))


class LumpedL2RieszMap:
    r"""The L2 Riesz map of a control space, with the mass lumped.

    A control built with it, ``Control(m, riesz_map=LumpedL2RieszMap(V))``,
    is measured in the lumped metric by everything pyadjoint does with it:
    the gradient ``derivative(apply_riesz=True)`` returns, and the gradient
    norm and initial Hessian of ``TAOSolver``. The map is diagonal, so it is
    applied as a coefficient-wise scaling.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space the control lives in.

    Attributes
    ----------
    mass : firedrake.Cofunction
        The lumped mass, one row sum per degree of freedom.
    """

    def __init__(self, function_space):
        self._function_space = function_space
        self.mass = _lumped_mass(function_space)

    def __call__(self, value):
        r"""Map a derivative to its gradient, or a field to its dual.

        Parameters
        ----------
        value : firedrake.Cofunction or firedrake.Function
            A dual object, mapped to :math:`M^{-1}` times it, or a primal
            one, mapped to :math:`M` times it.

        Returns
        -------
        firedrake.Function or firedrake.Cofunction
            The image of ``value``, in the space dual to its own.

        Raises
        ------
        ValueError
            If ``value`` is not in this map's space or its dual.
        """
        if ufl.duals.is_dual(value):
            if value.function_space().dual() != self._function_space:
                raise ValueError("Function space mismatch in LumpedL2RieszMap.")
            output = fire.Function(self._function_space)
            output.dat.data_wo[:] = value.dat.data_ro / self.mass.dat.data_ro
        elif ufl.duals.is_primal(value):
            if value.function_space() != self._function_space:
                raise ValueError("Function space mismatch in LumpedL2RieszMap.")
            output = fire.Cofunction(self._function_space.dual())
            output.dat.data_wo[:] = value.dat.data_ro * self.mass.dat.data_ro
        else:
            raise ValueError(
                f"Unable to ascertain if {value} is primal or dual."
            )
        return output


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


def minimize_with_tao(
    reduced_functional, bounds=None, comm=None, options=None, record=None,
):
    """Minimize a reduced functional with PETSc TAO.

    Under ensemble parallelism the controls are replicated on every member,
    so ``comm`` has to be the *spatial* communicator: TAO's default
    (``COMM_WORLD``) would count each member's copy as separate degrees of
    freedom.

    Parameters
    ----------
    reduced_functional : pyadjoint.ReducedFunctional
        Functional to minimize, and the controls to minimize it over.
    bounds : list of tuple, optional
        One ``(lower, upper)`` pair per control, each a scalar TAO broadcasts
        over the control or a value in the control's own space.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over.
    options : dict, optional
        PETSc options for the solver, such as ``{"tao_type": "blmvm",
        "tao_max_it": 20}``.
    record : callable, optional
        Called ``record(iteration, functional, controls)`` after each
        iteration TAO accepts, with the controls it stands at as a list of
        fresh fields. The starting point is not reported: it is the value the
        caller already has, from evaluating the functional to get here.

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
        If TAO stops without converging, which is what reaching the iteration
        limit amounts to. The last iterate is returned rather than raising,
        since a fixed iteration budget is a normal way to run an optimization.

    See Also
    --------
    LumpedL2RieszMap : The metric a bound-constrained inversion's controls
        should be built with, since TAO takes it from them.
    tao_bounds : Shapes ``vmin``/``vmax`` into the ``bounds`` this takes.
    """
    options = options or {}
    problem = MinimizationProblem(reduced_functional, bounds=bounds)
    solver = TAOSolver(problem, options, comm=comm)
    # TAO holds an iterate as one vector with every control concatenated into
    # it. Reading it through an interface built from those same controls lays
    # them out the way the solver's own does. Both the monitor and the
    # unconverged exit below need that, so it is built once.
    controls = [control.control for control in reduced_functional.controls]
    vec_interface = PETScVecInterface(tuple(controls), comm=comm)

    def iterate_of(tao):
        """Return the controls TAO currently stands at, as new fields."""
        iterate = [control.copy(deepcopy=True) for control in controls]
        vec_interface.from_petsc(tao.getSolution(), iterate)
        return iterate

    if record is not None:
        def monitor(tao):
            iteration, functional = tao.getSolutionStatus()[:2]
            if iteration:
                record(iteration, functional, iterate_of(tao))

        solver.tao.setMonitor(monitor)

    try:
        return as_list(solver.solve())
    except TAOConvergenceError as error:
        warnings.warn(
            f"{error} Returning the last iterate; raise the iteration limit "
            "or loosen the tolerances in the TAO options if the optimization "
            "is meant to run to convergence.",
        )
        return iterate_of(solver.tao)
