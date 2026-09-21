r"""Optimizers a reduced functional can be handed to.

What lives here is the optimizer's half of an inversion: shaping bounds into
the form TAO takes them, driving TAO, and reading the iterate back when it
stops short. None of it knows what is being inverted for.

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
:class:`LumpedTAOSolver` is for, and it is the only solver here:
:func:`minimize_with_tao` always uses it, because BLMVM is the only TAO type
this supports.

This module is built on pyadjoint's TAO support, some of it internal, so
``spyro.solvers.inversion`` imports it where it drives an optimization rather
than at the top of the file. An inversion that never asks for the automated
adjoint therefore never loads this, and never depends on what it needs.
"""

import warnings

import firedrake as fire
from mpi4py import MPI
import numpy as np

from pyadjoint import MinimizationProblem
from pyadjoint.enlisting import Enlist
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface,
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


def _lumped_initial_hessian(inverse_mass, vec_interface, comm):
    """Wrap the lumped Riesz map as the initial Hessian BLMVM starts from.

    TAO takes ``H0`` as a matrix it *solves* with, through a KSP of its own,
    so what is handed over is a matrix that is never applied, together with a
    preconditioner that applies its inverse -- the lumped Riesz map -- in a
    single ``preonly`` step. This mirrors how pyadjoint's ``TAOSolver`` seeds
    the same method with the consistent map.

    Parameters
    ----------
    inverse_mass : petsc4py.PETSc.Mat
        The lumped inverse Riesz map, from :func:`_lumped_riesz_map`.
    vec_interface : pyadjoint.optimization.tao_solver.PETScVecInterface
        Layout of the controls concatenated into one vector, which sizes the
        matrix.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm
        Communicator the controls are defined over.

    Returns
    -------
    tuple of (petsc4py.PETSc.Mat, petsc4py.PETSc.PC)
        The initial Hessian, and the preconditioner applying its inverse.
    """
    from petsc4py import PETSc

    class InitialHessian:
        """Context of the matrix TAO solves with, which is never applied."""

    class InitialHessianInverse:
        """Preconditioner context applying the lumped Riesz map."""

        def apply(self, pc, x, y):
            inverse_mass.mult(x, y)

    local_size, global_size = vec_interface.n, vec_interface.N
    matrix = PETSc.Mat().createPython(
        ((local_size, global_size), (local_size, global_size)),
        InitialHessian(),
        comm=comm,
    )
    matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    matrix.setUp()

    preconditioner = PETSc.PC().createPython(InitialHessianInverse(), comm=comm)
    preconditioner.setOperators(matrix)
    preconditioner.setUp()
    return matrix, preconditioner


class LumpedTAOSolver(OptimizationSolver):
    """TAO BLMVM with a diagonal metric, for box-constrained inversions.

    A near-copy of :class:`pyadjoint.TAOSolver`, differing in one decision
    that matters once the controls are bounded -- which Riesz map to use --
    and which enters in two places:

    The metric
        pyadjoint measures gradients in the consistent Riesz map. This uses
        the lumped one instead, so that the metric is coefficient-wise and
        agrees with TAO's coefficient-wise projection onto the box. See the
        module docstring for why the consistent map and the projection fight
        each other.

    The initial Hessian
        BLMVM approximates the inverse Hessian from the gradients it has
        seen, and starts that approximation from an ``H0``, which is also
        what turns the first derivative into a direction: the first step is
        along :math:`-H_0^{-1} DJ`. Like pyadjoint, this seeds ``H0`` with
        the Riesz map -- the lumped one again, so that the direction is the
        gradient of the module docstring, measured in the same metric as the
        convergence test. Left to PETSc, ``H0`` would be a scaled identity
        and the first direction the derivative read as a vector of
        coefficients, which differs from the gradient by the nodal masses.
        On spectral elements those vary by nearly two orders of magnitude
        within one element, and an optimizer stepping along the raw
        derivative moves the interior nodes and leaves the element edges
        behind.

    Only ``tao_type="blmvm"`` is supported: the lumped metric is there to
    serve the bound projection, and a solver that does not project has no use
    for it. The type is checked after the PETSc options are applied, so a
    type set through ``tao_options`` or the PETSc command line is caught
    rather than silently run with a metric meant for something else.

    No Hessian *callback* is registered, which is a separate thing from
    that initial Hessian: ``H0`` seeds an approximation BLMVM builds
    itself, whereas the callback would hand it a true second derivative, and
    a quasi-Newton method never asks for one. Measured over a run of this
    driver, the only callback TAO invokes is the combined
    objective-and-gradient one; the separate two are registered as well
    because a different line search may ask for them on their own.

    Parameters
    ----------
    problem : pyadjoint.MinimizationProblem
        The functional to minimize, its controls, and their bounds.
    parameters : dict
        PETSc options for the solver.
    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm, optional
        Communicator the controls are defined over. Under ensemble
        parallelism this is the *spatial* one.
    gradient_masks : list of firedrake.Function, optional
        One mask per control, in the control's own space, that every
        derivative handed to TAO is multiplied by, coefficient by
        coefficient. A coefficient whose mask is zero is never moved: its
        derivative is zero, so the direction is zero there under the
        diagonal metric, and the quasi-Newton pairs built from those
        directions keep it so. ``None`` applies no mask.

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
    minimize_with_tao : Drives this solver.
    """

    def __init__(self, problem, parameters, *, comm=None, gradient_masks=None):
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
        if gradient_masks is not None:
            gradient_masks = as_list(gradient_masks)
            if len(gradient_masks) != len(reduced_functional.controls):
                raise ValueError(
                    f"{len(reduced_functional.controls)} controls are being "
                    "optimized, so the gradient masks take that many "
                    f"entries; {len(gradient_masks)} were given.",
                )
        tao = PETSc.TAO().create(comm=comm)

        def masked(derivative):
            """Zero the derivative where the mask is zero, in place."""
            if gradient_masks is not None:
                for value, mask in zip(Enlist(derivative), gradient_masks):
                    value.dat.data[:] *= mask.dat.data_ro
            return derivative

        def objective(tao_, x):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            return tao_objective.objective(controls)

        def gradient(tao_, x, g):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            derivative = masked(tao_objective.gradient(controls))
            vec_interface.to_petsc(g, derivative)

        def objective_gradient(tao_, x, g):
            controls = new_control_variable(reduced_functional)
            vec_interface.from_petsc(x, controls)
            value, derivative = tao_objective.objective_gradient(controls)
            vec_interface.to_petsc(g, masked(derivative))
            return value

        tao.setObjective(objective)
        tao.setGradient(gradient)
        tao.setObjectiveGradient(objective_gradient)

        # No Hessian is set: BLMVM builds its own limited-memory metric from
        # the gradients it has seen, and never asks for one.
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
            tao, parameters=parameters, default_prefix="pyadjoint",
        )
        if tao.getType() != PETSc.TAO.Type.BLMVM:
            raise ValueError(
                "LumpedTAOSolver is restricted to tao_type='blmvm'."
            )

        # The first direction is -H0^{-1} DJ: seeded with the lumped mass,
        # it is the gradient the metric above measures, rather than the
        # derivative read as a vector of coefficients. See the class
        # docstring for what the difference does on spectral elements.
        initial_hessian, initial_hessian_inverse = _lumped_initial_hessian(
            inverse_mass, vec_interface, comm,
        )
        tao.setLMVMH0(initial_hessian)
        ksp = tao.getLMVMH0KSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
        ksp.setPC(initial_hessian_inverse)
        ksp.setUp()

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
        # Referenced by TAO's KSP; held here so they outlive this scope for
        # as long as the solver does.
        self._initial_hessian = initial_hessian
        self._initial_hessian_inverse = initial_hessian_inverse

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


def acquisition_mask(function_space, points, thickness):
    """Return the mask that zeroes the layers the sources and receivers sit in.

    A gradient is largest around the sources and the receivers, where the
    wavefields are strongest, and what it carries there is the imprint of
    the acquisition rather than information on the medium; zeroing it in a
    layer around them is usual practice in FWI [Modrak2016]_. The mask is
    one where the model may change and zero in every layer within
    ``thickness`` of the depth of a source or receiver: for an acquisition
    at the top of the domain, the layer below the surface; for a line of
    receivers at depth, the layer around it. The depth is the first mesh
    coordinate, in two and three dimensions alike.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Scalar space the mask is built in, that of the control it will
        multiply.
    points : array_like
        Positions of the sources and receivers, one per row, the depth
        first.
    thickness : float
        Half-thickness of the layers, in the mesh's units: a node closer
        than this to the depth of a point is zeroed.

    Returns
    -------
    firedrake.Function
        The mask, zero or one at every node of the space.

    References
    ----------
    .. [Modrak2016] Modrak, R., & Tromp, J. (2016). Seismic waveform
       inversion best practices: regional, global and exploration test
       cases. Geophysical Journal International, 206(3), 1864-1889.
    """
    depths = np.unique(np.atleast_2d(np.asarray(points, dtype=float))[:, 0])
    mesh = function_space.mesh()
    depth = fire.Function(function_space).interpolate(fire.SpatialCoordinate(mesh)[0])
    distance = np.abs(depth.dat.data_ro[:, None] - depths[None, :]).min(axis=1)
    mask = fire.Function(function_space, name="gradient_mask")
    mask.dat.data[:] = np.where(distance < thickness, 0.0, 1.0)
    return mask


def _acquisition_masks(controls, wave, radius, comm):
    """Build the masks of :func:`acquisition_mask`, one per control.

    Parameters
    ----------
    controls : list of firedrake.Function
        The controls TAO is given; one mask is built in the space of each,
        since the controls of an elastic inversion need not share one.
    wave : Wave
        The solver, for its sources, receivers, peak frequency and, if the
        radius is not given, its slowest wave speed.
    radius : float or None
        Half-thickness of the layers, in the mesh's units. ``None`` takes
        half the shortest wavelength of the current model at the peak
        frequency, the same on every rank of ``comm``.
    comm : mpi4py.MPI.Comm
        Communicator the controls are defined over.

    Returns
    -------
    list of firedrake.Function
        One mask per control.

    Raises
    ------
    ValueError
        If ``radius`` is negative.
    """
    if radius is None:
        # The slowest wave sets the shortest wavelength: the S wave of an
        # elastic medium, the only wave of an acoustic one.
        speed = wave.c if getattr(wave, "c_s", None) is None else wave.c_s
        c_min = comm.allreduce(
            float(speed.dat.data_ro.min()) if speed.dat.data_ro.size else np.inf,
            op=MPI.MIN,
        )
        radius = 0.5 * c_min / wave.frequency
    elif radius < 0:
        raise ValueError(f"mask_radius is a distance; received {radius}.")
    points = np.vstack([
        np.atleast_2d(np.asarray(wave.source_locations, dtype=float)),
        np.atleast_2d(np.asarray(wave.receiver_locations, dtype=float)),
    ])
    if comm.rank == 0:
        print(
            f"Zeroing the gradient within {radius:.4g} of the depth of every "
            "source and receiver (sources_receivers_gradient_mask).",
            flush=True,
        )
    return [
        acquisition_mask(control.function_space(), points, radius)
        for control in controls
    ]


def minimize_with_tao(
    reduced_functional, bounds=None, comm=None, options=None, record=None,
    gradient_masks=None, wave=None,
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
        PETSc options for the solver, such as ``{"tao_max_it": 20}``. The
        type has to resolve to ``blmvm``, which is the only one
        :class:`LumpedTAOSolver` supports; anything else raises there. Two
        options are spyro's own and are taken out before PETSc reads the
        rest: ``sources_receivers_gradient_mask``, whether to zero the
        gradient in a layer around the depth of every source and receiver
        (default False; see :func:`acquisition_mask`), and ``mask_radius``,
        the half-thickness of those layers in the mesh's units, half the
        shortest wavelength at the peak frequency by default. They need
        ``wave``.
    record : callable, optional
        Called ``record(iteration, functional, controls)`` after each
        iteration TAO accepts, with the controls it stands at as a list of
        fresh fields. The starting point is not reported: it is the value the
        caller already has, from evaluating the functional to get here.
    gradient_masks : list of firedrake.Function, optional
        One mask per control that every derivative handed to TAO is
        multiplied by, on top of the acquisition mask if that is on; see
        :class:`LumpedTAOSolver`.
    wave : Wave, optional
        The solver whose sources and receivers the acquisition mask is
        built around; needed only for that.

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

    Raises
    ------
    ValueError
        If the TAO type resolves to anything other than BLMVM, or the
        acquisition mask is asked for without ``wave``, or ``mask_radius``
        is given with it off.

    See Also
    --------
    LumpedTAOSolver : The solver this drives, and why its metric is lumped.
    tao_bounds : Shapes ``vmin``/``vmax`` into the ``bounds`` this takes.
    """
    options = dict(options or {})
    mask_acquisition = options.pop("sources_receivers_gradient_mask", False)
    mask_radius = options.pop("mask_radius", None)
    if mask_radius is not None and not mask_acquisition:
        raise ValueError(
            "mask_radius sizes the layers of sources_receivers_gradient_mask, "
            "which is off.",
        )
    if mask_acquisition:
        if wave is None:
            raise ValueError(
                "sources_receivers_gradient_mask needs the wave solver, for "
                "its sources and receivers.",
            )
        masks = _acquisition_masks(
            [control.control for control in reduced_functional.controls],
            wave, mask_radius, comm,
        )
        if gradient_masks is not None:
            for mask, given in zip(masks, gradient_masks):
                mask.dat.data[:] *= given.dat.data_ro
        gradient_masks = masks
    problem = MinimizationProblem(reduced_functional, bounds=bounds)
    solver = LumpedTAOSolver(
        problem, options, comm=comm, gradient_masks=gradient_masks,
    )
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
