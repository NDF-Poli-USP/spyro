r"""Optimizers a reduced functional can be handed to.

What lives here is the optimizer's half of an inversion: shaping bounds into
the form TAO takes them, driving TAO, and reading the iterate back when it
stops short. None of it knows what is being inverted for.

The metric
----------
A gradient computed by the adjoint is a *dual* object: it lives in
:math:`V'`, and turning it into a direction in :math:`V` takes the Riesz map,
:math:`\nabla J = M^{-1} DJ` with :math:`M` the mass matrix. TAO is told which
metric to measure gradients in through ``setGradientNorm``.

For a *bound-constrained* problem that choice is not free. TAO projects onto
the box coefficient by coefficient, and a projection is only a projection in a
metric that is itself coefficient-wise. A consistent mass matrix couples
neighbouring degrees of freedom, so the projected point is not the closest
feasible point in that metric, and the two fight each other. Lumping the mass
-- collapsing it to its row sums, which is a diagonal matrix -- makes the
metric coefficient-wise too, and the two agree again. That is what
:class:`LumpedTAOSolver` is for; :func:`minimize_with_tao` selects it whenever
BLMVM is the requested type.
"""

import warnings

import firedrake as fire
import numpy as np

from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.enlisting import Enlist
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface,
    ReducedFunctionalMat,
    RFOperation,
    TAOConvergenceError,
    TAOObjective,
    _tao_reasons,
    new_control_variable,
    valid_comm,
)

from ..domains.quadrature import quadrature_rules
from ..utils.physical_parameters import as_list


def _inverse_lumped_mass(function_space):
    r"""Return the inverse lumped mass of a control space, as a PETSc vector.

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
    firedrake.Function
        The reciprocal of each row sum. A field rather than a bare vector so
        that ``PETScVecInterface.to_petsc`` can lay it out, which is what
        keeps this out of the interface's private index sets.
    """
    trial = fire.TrialFunction(function_space)
    test = fire.TestFunction(function_space)
    one = fire.Function(function_space).assign(1.0)

    try:
        quadrature, _, _ = quadrature_rules(function_space)
    except ValueError:
        quadrature = {}

    measure = fire.dx(**quadrature) if quadrature else fire.dx
    diagonal = fire.assemble(fire.action(trial * test * measure, one))

    inverse = fire.Function(function_space)
    with diagonal.dat.vec_ro as lumped, inverse.dat.vec_wo as target:
        lumped.copy(target)
        target.reciprocal()
    return inverse


class _LumpedRieszMapContext:
    r"""Python-matrix context applying the lumped inverse Riesz map.

    The matrix it backs takes a dual object -- a derivative -- and returns the
    primal direction it corresponds to, :math:`\nabla J = M^{-1} DJ`, with
    :math:`M` lumped so that the map is a coefficient-wise scaling.

    An optimization over more than one control gets one block per control:
    they may live in different spaces, so each carries its own lumped mass.
    The blocks are addressed through the index sets of the same
    ``PETScVecInterface`` layout TAO is given, which is what keeps the
    concatenated vector's pieces matched to the controls they belong to.

    Parameters
    ----------
    controls : pyadjoint.Control or list of pyadjoint.Control
        Controls being optimized, in the order TAO holds them.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over.

    Attributes
    ----------
    controls : pyadjoint.enlisting.Enlist
        The controls, as a list however many there are.
    vec_interface : pyadjoint.optimization.tao_solver.PETScVecInterface
        Layout of the controls concatenated into one vector.
    inverse_mass : petsc4py.PETSc.Vec
        Every control's inverse lumped mass, concatenated into the layout the
        interface defines. The map is diagonal, so one vector is the whole of
        it and applying it is a single pointwise product.
    """

    def __init__(self, controls, comm=None):
        comm = valid_comm(comm)
        self.controls = Enlist(controls)
        self.vec_interface = PETScVecInterface(
            tuple(control.control for control in self.controls), comm=comm,
        )
        self.inverse_mass = self.vec_interface.new_petsc()
        self.vec_interface.to_petsc(self.inverse_mass, [
            _inverse_lumped_mass(control.control.function_space())
            for control in self.controls
        ])

    def mult(self, mat, x, y):
        """Scale each control's block of ``x`` by its inverse lumped mass.

        Parameters
        ----------
        mat : petsc4py.PETSc.Mat
            The matrix this context backs. Unused: the map carries no state
            beyond the masses.
        x : petsc4py.PETSc.Vec
            Dual values, the controls concatenated.
        y : petsc4py.PETSc.Vec
            Where the primal values are written, laid out the same way.

        Returns
        -------
        None
            Written into ``y`` in place.
        """
        y.pointwiseMult(x, self.inverse_mass)


def _lumped_riesz_map(controls, comm=None):
    """Build the PETSc matrix TAO measures gradients with.

    A Python matrix over :class:`_LumpedRieszMapContext`: nothing is stored
    beyond one vector per control, since the map is diagonal. It is declared
    symmetric because it is, and TAO uses that.

    Parameters
    ----------
    controls : pyadjoint.Control or list of pyadjoint.Control
        Controls being optimized, in the order TAO holds them.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over.

    Returns
    -------
    petsc4py.PETSc.Mat
        The inverse Riesz map, to hand to ``TAO.setGradientNorm``.
    """
    from petsc4py import PETSc

    context = _LumpedRieszMapContext(controls, comm=comm)
    local_size = context.vec_interface.n
    global_size = context.vec_interface.N
    matrix = PETSc.Mat().createPython(
        ((local_size, global_size), (local_size, global_size)),
        context,
        comm=context.vec_interface.comm,
    )
    matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    matrix.setUp()
    matrix.assemble()
    return matrix


class LumpedTAOSolver(OptimizationSolver):
    """TAO BLMVM with a diagonal metric, for box-constrained inversions.

    A near-copy of :class:`pyadjoint.TAOSolver`, differing in two decisions
    that matter once the controls are bounded:

    The metric
        pyadjoint measures gradients in the consistent Riesz map. This uses
        the lumped one instead, so that the metric is coefficient-wise and
        agrees with TAO's coefficient-wise projection onto the box. See the
        module docstring for why the consistent map and the projection fight
        each other.

    The initial Hessian
        pyadjoint injects the Riesz map as LMVM's ``H0``, fixing the scale of
        the first quasi-Newton step. That is not done here, so PETSc keeps its
        own dynamic scaling. It is the reason the two solvers take different
        paths from the same starting point, even where both converge.

    Restricted to ``tao_type="blmvm"``: the lumped metric is there to serve
    the bound projection, and a solver that does not project has no use for
    it. The restriction is checked after the PETSc options are applied, so it
    also catches a type set from the command line.

    Parameters
    ----------
    problem : pyadjoint.MinimizationProblem
        The functional to minimize, its controls, and their bounds.
    parameters : dict
        PETSc options for the solver.
    options_prefix : str, optional
        Prefix for this solver's PETSc options.
    appctx : dict, optional
        User context handed to the Hessian action.
    Pmat : petsc4py.PETSc.Mat, optional
        Preconditioner for the Hessian. Defaults to the Hessian itself.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over. Under ensemble
        parallelism this is the *spatial* one.

    Raises
    ------
    TypeError
        If ``problem`` is not a :class:`pyadjoint.MinimizationProblem`.
    NotImplementedError
        If the problem carries constraints, which TAO is not set up for here.
    ValueError
        If the resolved TAO type is not BLMVM.

    See Also
    --------
    minimize_with_tao : Selects this solver whenever BLMVM is asked for.
    """

    def __init__(
        self,
        problem,
        parameters,
        *,
        options_prefix=None,
        appctx=None,
        Pmat=None,
        comm=None,
    ):
        from petsc4py import PETSc
        import petsctools

        if not isinstance(problem, MinimizationProblem):
            raise TypeError("MinimizationProblem required")
        if problem.constraints is not None:
            raise NotImplementedError("Constraints not implemented")

        comm = valid_comm(comm)
        reduced_functional = problem.reduced_functional
        tao_objective = TAOObjective(reduced_functional)
        vec_interface = PETScVecInterface(
            tuple(
                control.control for control in reduced_functional.controls
            ),
            comm=comm,
        )
        tao = PETSc.TAO().create(comm=comm)

        def objective(tao_, x):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            return tao_objective.objective(controls)

        def gradient(tao_, x, g):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            derivative = tao_objective.gradient(controls)
            vec_interface.to_petsc(g, derivative)

        def objective_gradient(tao_, x, g):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            value, derivative = tao_objective.objective_gradient(controls)
            vec_interface.to_petsc(g, derivative)
            return value

        tao.setObjective(objective)
        tao.setGradient(gradient)
        tao.setObjectiveGradient(objective_gradient)

        hessian = ReducedFunctionalMat(
            reduced_functional,
            appctx=appctx,
            action=RFOperation.HESSIAN,
            comm=comm,
        )
        tao.setHessian(
            hessian.getPythonContext().update,
            H=hessian,
            P=Pmat or hessian,
        )

        inverse_mass = _lumped_riesz_map(
            reduced_functional.controls, comm=comm,
        )
        tao.setGradientNorm(inverse_mass)

        if problem.bounds is not None:
            lower_bounds = []
            upper_bounds = []
            for lower, upper in problem.bounds:
                lower_bounds.append(
                    np.finfo(PETSc.ScalarType).min
                    if lower is None else lower
                )
                upper_bounds.append(
                    np.finfo(PETSc.ScalarType).max
                    if upper is None else upper
                )
            lower_vector = vec_interface.new_petsc()
            upper_vector = vec_interface.new_petsc()
            vec_interface.to_petsc(lower_vector, lower_bounds)
            vec_interface.to_petsc(upper_vector, upper_bounds)
            tao.setVariableBounds(lower_vector, upper_vector)

        petsctools.set_from_options(
            tao,
            parameters=parameters,
            options_prefix=options_prefix,
            default_prefix="pyadjoint",
        )
        if tao.getType() != PETSc.TAO.Type.BLMVM:
            raise ValueError(
                "LumpedTAOSolver is restricted to tao_type='blmvm'."
            )

        solution = vec_interface.new_petsc()
        tao.setSolution(solution)
        with petsctools.inserted_options(tao):
            tao.setUp()

        super().__init__(problem, parameters)
        self._tao_objective = tao_objective
        self._vec_interface = vec_interface
        self._tao = tao
        self._x = solution
        self._inverse_mass = inverse_mass

    @property
    def tao_objective(self):
        """:class:`pyadjoint.optimization.tao_solver.TAOObjective`: what TAO \
        evaluates, wrapping the reduced functional."""
        return self._tao_objective

    @property
    def tao(self):
        """:class:`petsc4py.PETSc.TAO`: the solver itself, for a monitor or \
        for reading its state after a run."""
        return self._tao

    @property
    def x(self):
        """:class:`petsc4py.PETSc.Vec`: the solution vector, every control \
        concatenated into it. Holds the last iterate even when the solve \
        stops short."""
        return self._x

    def solve(self):
        """Run BLMVM from the controls' current values.

        Returns
        -------
        OverloadedType or tuple of OverloadedType
            The controls TAO converged on, shaped the way pyadjoint returns
            them: a bare one for a single control, a tuple for several.

        Raises
        ------
        pyadjoint.optimization.tao_solver.TAOConvergenceError
            If TAO stops for any reason other than convergence, the iteration
            limit included. :func:`minimize_with_tao` catches this and reads
            the last iterate out of :attr:`x`.
        """
        import petsctools

        controls = self.tao_objective.reduced_functional.controls
        values = tuple(control.tape_value()._ad_copy() for control in controls)
        with petsctools.inserted_options(self.tao):
            self._vec_interface.to_petsc(self.x, values)
            self.tao.solve()
            self._vec_interface.from_petsc(self.x, values)

        reason = self.tao.getConvergedReason()
        if reason <= 0:
            # Named rather than numbered: "DIVERGED_MAXITS" tells the caller
            # to raise the iteration limit, "-2" tells them nothing.
            raise TAOConvergenceError(
                "LumpedTAOSolver failed to converge after "
                f"{self.tao.getIterationNumber()} iterations with reason: "
                f"{_tao_reasons.get(reason, reason)}."
            )
        if isinstance(controls, Enlist):
            return controls.delist(values)
        return values


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
        PETSc options for the solver, such as ``{"tao_type": "blmvm"}``.
        BLMVM is driven by :class:`LumpedTAOSolver`, which measures gradients
        in the lumped metric its box projection needs; every other type is
        left to pyadjoint's own :class:`pyadjoint.TAOSolver`. The choice is
        read from this mapping, so a type set only through the PETSc command
        line reaches the solver but not this dispatch.
    record : callable, optional
        Called ``record(iteration, functional)`` after each iteration TAO
        accepts. The starting point is not reported: it is the value the
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
    LumpedTAOSolver : The BLMVM driver, and why its metric is lumped.
    tao_bounds : Shapes ``vmin``/``vmax`` into the ``bounds`` this takes.
    """
    options = options or {}
    problem = MinimizationProblem(reduced_functional, bounds=bounds)
    solver_type = (
        LumpedTAOSolver
        if str(options.get("tao_type", "")).lower() == "blmvm"
        else TAOSolver
    )
    solver = solver_type(problem, options, comm=comm)
    if record is not None:
        def monitor(tao):
            iteration, functional = tao.getSolutionStatus()[:2]
            if iteration:
                record(iteration, functional)

        solver.tao.setMonitor(monitor)

    try:
        return as_list(solver.solve())
    except TAOConvergenceError as error:
        warnings.warn(
            f"{error} Returning the last iterate; raise the iteration limit "
            "or loosen the tolerances in the TAO options if the optimization "
            "is meant to run to convergence.",
        )
        # TAO raises before handing the iterate back, and holds it as one
        # vector with every control concatenated into it. Reading it through
        # an interface built from those same controls lays them out the way
        # the solver's own does.
        controls = [control.control for control in reduced_functional.controls]
        iterate = [control.copy(deepcopy=True) for control in controls]
        PETScVecInterface(tuple(controls), comm=comm).from_petsc(
            solver.tao.getSolution(), iterate,
        )
        return iterate
