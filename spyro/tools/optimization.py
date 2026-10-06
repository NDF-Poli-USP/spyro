r"""Optimizers a reduced functional can be handed to.

Tools to minimize a reduced functional with pyadjoint's ``TAOSolver``: the
metric of the controls, the bounds in the form TAO expects, and the call to
TAO itself.

The metric
----------
The adjoint gives the derivative :math:`DJ \in V^{\ast}`. The gradient is its
Riesz representer, :math:`\nabla J = M^{-1} DJ`, with :math:`M` the mass
matrix. TAO uses this metric for its steps and for its convergence test
(``tao_gatol``, ``tao_grtol``), which makes the tolerances independent of the
mesh.

With bounds, the metric must be diagonal: TAO projects onto the box one
coefficient at a time, which is the true projection only for a diagonal
metric. The mass is therefore lumped, :math:`M_L`.

Passing :math:`M_L^{-1}` to TAO as the initial Hessian would stop PETSc from
rescaling the quasi-Newton step. Instead,
:class:`LumpedL2TransformedFunctional` changes variables to
:math:`z = M_L^{1/2} m`. In :math:`z` the lumped :math:`L^2` inner product is
the Euclidean one, :math:`m^T M_L m = z^T z`, so TAO's standard method, with
its own step rescaling, is the :math:`L^2` method in :math:`m`. Since
:math:`M_L` is diagonal, bounds on :math:`m` remain bounds on :math:`z`.

The default method is BQNLS. For LMVM and BLMVM, pyadjoint sets a fixed
initial Hessian that disables PETSc's rescaling; BLMVM also restarts each line
search with a unit step, and LMVM ignores bounds. :func:`minimize_with_tao`
warns if either is used.

This module is imported only when the automated adjoint drives an
optimization, so other inversions do not depend on pyadjoint's TAO support.
"""

import warnings

import firedrake as fire
import numpy as np

from pyadjoint import Control, MinimizationProblem, TAOSolver, no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface,
    TAOConvergenceError,
)
from pyadjoint.reduced_functional import AbstractReducedFunctional

from ..domains.quadrature import quadrature_rules
from ..utils.physical_parameters import as_list


@no_annotations
def _lumped_mass(function_space: fire.FunctionSpace) -> fire.Cofunction:
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

    Raises
    ------
    ValueError
        If a row sum is not positive. A row sum is the integral of a basis
        function, and some Lagrange elements have basis functions that
        integrate to zero or less: the vertex functions of quadratic Lagrange
        integrate to zero on triangles and to a negative value on
        tetrahedra. The lumped metric is then not positive definite, so it
        cannot be the metric TAO runs in.
    """
    trial = fire.TrialFunction(function_space)
    test = fire.TestFunction(function_space)
    one = fire.Function(function_space).assign(1.0)

    try:
        quadrature, _, _ = quadrature_rules(function_space)
    except ValueError:
        quadrature = {}

    measure = fire.dx(**quadrature) if quadrature else fire.dx
    mass = fire.assemble(fire.action(trial * test * measure, one))
    with mass.dat.vec_ro as entries:
        _, smallest = entries.min()
    if smallest <= 0.0:
        element = function_space.ufl_element()
        raise ValueError(
            "The lumped mass of a control space has to be positive, and this "
            f"one, {element.family()} of degree {element.degree()} on a "
            f"{function_space.mesh().ufl_cell()} mesh, has an entry "
            f"of {smallest:.3g}. Lumping by row sums gives each degree of "
            "freedom the integral of its basis function, and some basis "
            "functions of this element integrate to zero or less (the vertex "
            "functions of quadratic Lagrange do, on triangles and "
            "tetrahedra), so the lumped metric TAO would run in is not "
            "positive definite. Use a mass-lumped element for the controls "
            "(KMV, or spectral on quadrilaterals), or a Lagrange degree whose "
            "basis functions all integrate to a positive value, such as 1.",
        )
    return mass


class LumpedL2TransformedFunctional(AbstractReducedFunctional):
    r"""A reduced functional over controls rescaled by their lumped mass.

    Represents :math:`\hat{J}(z) = J(M_L^{-1/2} z)`: each control :math:`m`
    of the wrapped functional is replaced by :math:`z = M_L^{1/2} m`, with
    :math:`M_L` its lumped mass. See the module docstring for why.

    The controls here carry the ``"l2"`` Riesz map, which in :math:`z` is the
    lumped :math:`L^2` one in :math:`m`. Values go in and come out as
    :math:`z`; :meth:`map_result` takes them back to :math:`m`, and
    :meth:`transform_bounds` takes bounds on :math:`m` to bounds on
    :math:`z`.

    The diagonal analogue of :class:`firedrake.adjoint.L2TransformedFunctional`,
    which keeps the consistent mass through a block-diagonal factorization in
    a DG space. That turns a box on :math:`m` into a general polytope, which
    TAO cannot project onto; a diagonal transformation keeps it a box.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional of the model controls.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional):
        super().__init__()
        self._functional = reduced_functional
        model_controls = reduced_functional.controls
        if not isinstance(model_controls, Enlist):
            model_controls = Enlist(model_controls)
        self._model_controls = model_controls

        masses = {}
        self._scales = []
        transformed = []
        for control in model_controls:
            model = control.control
            space = model.function_space()
            if space not in masses:
                masses[space] = _lumped_mass(space)
            # m = scale * z, with scale the inverse square root of the mass.
            scale = fire.Function(space)
            scale.dat.data_wo[:] = 1.0 / np.sqrt(masses[space].dat.data_ro)
            self._scales.append(scale)
            z = fire.Function(space)
            z.dat.data_wo[:] = model.dat.data_ro / scale.dat.data_ro
            transformed.append(Control(z, riesz_map="l2"))
        self._controls = Enlist(model_controls.delist(transformed))

    @property
    def controls(self) -> Enlist:
        """:class:`pyadjoint.enlisting.Enlist`: the controls over :math:`z`."""
        return self._controls

    def _scaled(self, values, dual: bool) -> list:
        """Multiply each value by its control's scale, coefficient-wise.

        Parameters
        ----------
        values : firedrake.Function, firedrake.Cofunction or sequence
            One value per control.
        dual : bool
            Whether the values are Cofunctions.

        Returns
        -------
        list
            The scaled values, new objects of the same kind.
        """
        scaled = []
        for value, scale in zip(Enlist(values), self._scales):
            space = scale.function_space()
            out = fire.Cofunction(space.dual()) if dual else fire.Function(space)
            out.dat.data_wo[:] = value.dat.data_ro * scale.dat.data_ro
            scaled.append(out)
        return scaled

    def __call__(self, values):
        """Evaluate the functional at transformed control values.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            Values of :math:`z`, one per control.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional at :math:`m = M_L^{-1/2} z`.
        """
        models = self._scaled(values, dual=False)
        return self._functional(self._model_controls.delist(models))

    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        """Return the derivative with respect to :math:`z`.

        By the chain rule it is the derivative with respect to :math:`m`,
        scaled coefficient-wise by :math:`M_L^{-1/2}`.

        Parameters
        ----------
        adj_input : float, optional
            Adjoint value of the functional result.
        apply_riesz : bool, optional
            Whether to return the gradient, through the ``"l2"`` Riesz map,
            instead of the derivative.

        Returns
        -------
        firedrake.Cofunction, firedrake.Function or list
            One per control, shaped like the controls.
        """
        derivative = self._functional.derivative(adj_input=adj_input)
        scaled = self._scaled(derivative, dual=True)
        if apply_riesz:
            scaled = [
                control.control._ad_convert_riesz(value, riesz_map="l2")
                for control, value in zip(self._controls, scaled)
            ]
        return self._controls.delist(scaled)

    def tlm(self, m_dot):
        """Return the tangent linear action along a direction in :math:`z`.

        Parameters
        ----------
        m_dot : firedrake.Function or sequence of firedrake.Function
            Direction in :math:`z`, one per control.

        Returns
        -------
        pyadjoint.OverloadedType
            The tangent linear action, of the functional's type.
        """
        directions = self._scaled(m_dot, dual=False)
        return self._functional.tlm(self._model_controls.delist(directions))

    def hessian(self, m_dot, hessian_input=None, evaluate_tlm: bool = True,
                apply_riesz: bool = False):
        """Return the Hessian action along a direction in :math:`z`.

        Parameters
        ----------
        m_dot : firedrake.Function or sequence of firedrake.Function
            Direction in :math:`z`, one per control.
        hessian_input : pyadjoint.OverloadedType, optional
            Hessian value of the functional result.
        evaluate_tlm : bool, optional
            Whether to evaluate the tangent linear model first.
        apply_riesz : bool, optional
            Whether to map the result through the ``"l2"`` Riesz map.

        Returns
        -------
        firedrake.Cofunction, firedrake.Function or list
            One per control, shaped like the controls.
        """
        directions = self._scaled(m_dot, dual=False)
        action = self._functional.hessian(
            self._model_controls.delist(directions),
            hessian_input=hessian_input, evaluate_tlm=evaluate_tlm,
        )
        scaled = self._scaled(action, dual=True)
        if apply_riesz:
            scaled = [
                control.control._ad_convert_riesz(value, riesz_map="l2")
                for control, value in zip(self._controls, scaled)
            ]
        return self._controls.delist(scaled)

    def map_result(self, values) -> list:
        """Map values of :math:`z` back to the model controls.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            Values of :math:`z`, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`m = M_L^{-1/2} z`, one per control, named after the
            controls of the wrapped functional.
        """
        models = self._scaled(values, dual=False)
        for model, control in zip(models, self._model_controls):
            model.rename(control.control.name())
        return models

    def transform_bounds(self, bounds) -> list:
        """Map bounds on the model controls to bounds on :math:`z`.

        Parameters
        ----------
        bounds : sequence of tuple
            One ``(lower, upper)`` pair per control, each a scalar, a field
            in the control's space, or None.

        Returns
        -------
        list of tuple
            The same pairs on :math:`z`, each bound that is not None a field
            in the control's space.
        """
        transformed = []
        for (lower, upper), scale in zip(bounds, self._scales):
            pair = []
            for bound in (lower, upper):
                if bound is None:
                    pair.append(None)
                    continue
                field = fire.Function(scale.function_space())
                values = bound.dat.data_ro if isinstance(bound, fire.Function) else bound
                field.dat.data_wo[:] = values / scale.dat.data_ro
                pair.append(field)
            transformed.append(tuple(pair))
        return transformed


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
    reduced_functional: AbstractReducedFunctional,
    bounds: list | None = None,
    comm=None,
    options: dict | None = None,
    record=None,
) -> list:
    """Minimize a reduced functional with PETSc TAO, in the lumped L2 metric.

    The optimization runs over :class:`LumpedL2TransformedFunctional`, so TAO
    measures gradients, takes steps and projects onto the bounds in the
    lumped :math:`L^2` metric of the controls; see the module docstring.

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
        over the control or a value in the control's own space.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over.
    options : dict, optional
        PETSc options for the solver, such as ``{"tao_max_it": 20}``, merged
        over ``{"tao_type": "bqnls"}``. Without a type TAO would fall back on
        its own default, LMVM, which ignores bounds.
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
        since a fixed iteration limit is a normal way to run an optimization.
    UserWarning
        If the TAO type resolves to LMVM or BLMVM, with the reasons it is not
        recommended: pyadjoint's ``TAOSolver`` gives both an initial Hessian
        that PETSc keeps fixed instead of rescaling; BLMVM restarts every line
        search from a unit step; and LMVM ignores the bounds.

    See Also
    --------
    LumpedL2TransformedFunctional : The change of variables TAO runs in.
    tao_bounds : Shapes ``vmin``/``vmax`` into the ``bounds`` this takes.
    """
    options = {"tao_type": "bqnls", **(options or {})}
    transformed = LumpedL2TransformedFunctional(reduced_functional)
    if bounds is not None:
        bounds = transformed.transform_bounds(bounds)
    problem = MinimizationProblem(transformed, bounds=bounds)
    solver = TAOSolver(problem, options, comm=comm)
    # Checked on the type TAO resolved, which the PETSc command line can set
    # as well as ``options``.
    tao_type = solver.tao.getType()
    if tao_type in {"lmvm", "blmvm"}:
        reasons = [
            "pyadjoint's TAOSolver gives it a fixed initial Hessian, which "
            "turns off PETSc's rescaling of the quasi-Newton step",
        ]
        if tao_type == "blmvm":
            reasons.append(
                "it restarts every line search from a unit step, ignoring "
                "tao_ls_stepinit",
            )
        elif bounds is not None:
            reasons.append(
                "it is unconstrained, so it ignores the bounds and the "
                "controls can leave them",
            )
        warnings.warn(
            f"TAO type '{tao_type}' is not recommended here: "
            + "; ".join(reasons)
            + ". The default 'bqnls', which TAO introduced to replace "
            "BLMVM, has none of these problems.",
        )
    # TAO holds an iterate as one vector with every control concatenated into
    # it. Reading it through an interface built from those same controls lays
    # them out the way the solver's own does. Both the monitor and the
    # unconverged exit below need that, so it is built once.
    controls = [control.control for control in transformed.controls]
    vec_interface = PETScVecInterface(tuple(controls), comm=comm)

    def iterate_of(tao):
        """Return the model controls TAO currently stands at, as new fields."""
        iterate = [control.copy(deepcopy=True) for control in controls]
        vec_interface.from_petsc(tao.getSolution(), iterate)
        return transformed.map_result(iterate)

    if record is not None:
        def monitor(tao):
            iteration, functional = tao.getSolutionStatus()[:2]
            if iteration:
                record(iteration, functional, iterate_of(tao))

        solver.tao.setMonitor(monitor)

    try:
        return transformed.map_result(as_list(solver.solve()))
    except TAOConvergenceError as error:
        warnings.warn(
            f"{error} Returning the last iterate; raise the iteration limit "
            "or loosen the tolerances in the TAO options if the optimization "
            "is meant to run to convergence.",
        )
        return iterate_of(solver.tao)
