"""A reduced functional over controls rescaled by their lumped mass."""

import firedrake as fire
import numpy as np

from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional

from ..domains.quadrature import quadrature_rules


@no_annotations
def _lumped_mass(function_space: fire.FunctionSpace) -> fire.Function:
    r"""Return :math:`M_L`, the diagonal of the lumped mass matrix.

    See :class:`LumpedL2ReducedFunctional` for the quadrature it is assembled
    with.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space a control lives in.

    Returns
    -------
    firedrake.Function
        The diagonal of the mass matrix, one entry per degree of freedom.

    Raises
    ------
    ValueError
        If the mass matrix assembled with that quadrature is not diagonal.
    """
    from petsc4py import PETSc

    trial = fire.TrialFunction(function_space)
    test = fire.TestFunction(function_space)
    try:
        quadrature, _, _ = quadrature_rules(function_space)
    except ValueError:
        quadrature = {}
    measure = fire.dx(**quadrature) if quadrature else fire.dx
    mass = fire.assemble(trial * test * measure).petscmat

    diagonal = mass.getDiagonal()
    off_diagonal = mass.duplicate(copy=True)
    zeros = diagonal.duplicate()
    zeros.zeroEntries()
    off_diagonal.setDiagonal(zeros)
    largest = off_diagonal.norm(PETSc.NormType.INFINITY) / diagonal.max()[1]
    if largest > 1e-12:
        element = function_space.ufl_element()
        raise ValueError(
            "LumpedL2ReducedFunctional needs a control space whose mass "
            "matrix, assembled with the quadrature spyro adopts for the "
            "element, is diagonal: KMV elements, spectral (GLL) "
            "quadrilaterals or hexahedra, or DG0. This one, "
            f"{element.family()} of degree {element.degree()} on a "
            f"{function_space.mesh().ufl_cell()} mesh, is not: its "
            f"off-diagonal entries reach {largest:.2g} of the diagonal.",
        )

    lumped = fire.Function(function_space)
    with lumped.dat.vec_wo as values:
        diagonal.copy(values)
    return lumped


def _inverse_sqrt_lumped_mass(
    function_space: fire.FunctionSpace,
) -> fire.Function:
    r"""Return :math:`M_L^{-1/2}`, the inverse square root of the lumped mass.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space a control lives in.

    Returns
    -------
    firedrake.Function
        The scale taking :math:`\tilde{m}` to :math:`m = M_L^{-1/2} \tilde{m}`.

    Raises
    ------
    ValueError
        If the mass matrix of the space is not diagonal.
    """
    scale = _lumped_mass(function_space)
    with scale.dat.vec as values:
        values.sqrtabs()
        values.reciprocal()
    return scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic function of ``values``.

    Parameters
    ----------
    values : numpy.ndarray
        Latent values.

    Returns
    -------
    numpy.ndarray
        :math:`1 / (1 + e^{-\\psi})`, computed without overflow.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def _bound_array(bound, space: fire.FunctionSpace) -> np.ndarray:
    """Return a bound as values at the degrees of freedom of ``space``.

    Parameters
    ----------
    bound : float or firedrake.Function
        The bound.
    space : firedrake.FunctionSpace
        Space of the control it bounds.

    Returns
    -------
    numpy.ndarray
        One value per degree of freedom owned by this process.

    Raises
    ------
    ValueError
        If ``bound`` is None: the latent map needs both bounds.
    """
    if bound is None:
        raise ValueError("latent_bounds needs a lower and an upper bound "
                         "for every control.")
    if isinstance(bound, fire.Function):
        return bound.dat.data_ro.copy()
    return np.full(fire.Function(space).dat.data_ro.shape, float(bound))


def _box_bounds(bounds: list | tuple, controls: Enlist) -> list:
    """Validate finite box bounds and return their lower values and widths.

    Parameters
    ----------
    bounds : sequence of tuple
        One lower/upper pair per control, containing scalars or fields.
    controls : pyadjoint.enlisting.Enlist
        Controls whose spaces and communicators define the bounds.

    Returns
    -------
    list of tuple of numpy.ndarray
        Lower values and nonnegative widths on locally owned degrees of freedom.

    Raises
    ------
    ValueError
        If bounds are missing, nonfinite, reversed or have incompatible sizes.
    """
    if bounds is None or len(bounds) != len(controls):
        raise ValueError("bounds must contain one pair per control.")
    result = []
    for control, pair in zip(controls, bounds):
        if len(pair) != 2:
            raise ValueError("Each bound must be a (lower, upper) pair.")
        space = control.control.function_space()
        lower, upper = [_bound_array(bound, space) for bound in pair]
        shape = control.control.dat.data_ro.shape
        invalid = lower.shape != shape or upper.shape != shape
        if not invalid:
            invalid = (not np.all(np.isfinite(lower))
                       or not np.all(np.isfinite(upper))
                       or np.any(upper < lower)
                       or not np.all(np.isfinite(upper - lower)))
        if space.mesh().comm.allreduce(int(invalid)):
            raise ValueError("bounds must be finite, ordered and match the control.")
        result.append((lower, upper - lower))
    return result


class LumpedL2ReducedFunctional(AbstractReducedFunctional):
    r"""A reduced functional over controls rescaled by their lumped mass.

    Represents :math:`\hat{J}(\tilde{v}) = J(m)`. TAO works with
    :math:`\tilde{v} = M_L^{1/2} v`, with :math:`M_L` the lumped mass, and

    .. math::

        m = v, \qquad\text{or, with ``latent_bounds``,}\qquad
        m = \ell + (u - \ell)\,\sigma(v), \quad
        \sigma(v) = \frac{1}{1 + e^{-v}}.

    The Euclidean inner product in :math:`\tilde{v}` is the lumped
    :math:`L^2` one in :math:`v`, so TAO measures gradients and steps in that
    metric. With ``latent_bounds``, :math:`v` is the latent control
    :math:`\psi`: any :math:`\psi` gives a model inside :math:`[\ell, u]`,
    so TAO needs no bounds. Degrees of freedom with :math:`\ell = u` stay
    fixed.

    :math:`M_L` is the mass matrix assembled with the quadrature spyro adopts
    for each element (:func:`spyro.domains.quadrature.quadrature_rules`). For
    KMV elements and spectral (GLL) quadrilaterals the quadrature points are
    the nodes, so this matrix is diagonal at any degree. Control spaces where
    it is not diagonal are rejected.

    Values go in and come out as :math:`\tilde{v}`; :meth:`map_result`
    takes them to :math:`m`, and :meth:`transform_bounds` takes bounds on
    :math:`m` to bounds on :math:`\tilde{v}` when there are no
    ``latent_bounds``.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional of the model controls.
    latent_bounds : sequence of tuple, optional
        ``(lower, upper)`` on :math:`m`, one per control, each a scalar or a
        field. Given, the optimization runs over latent controls inside them.

    Raises
    ------
    ValueError
        If the mass matrix of a control space is not diagonal, or
        ``latent_bounds`` are missing, nonfinite or reversed.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional,
                 latent_bounds: list | tuple | None = None) -> None:
        super().__init__()
        self._functional = reduced_functional
        model_controls = reduced_functional.controls
        if not isinstance(model_controls, Enlist):
            model_controls = Enlist(model_controls)
        self._model_controls = model_controls
        # (lower, width) per control, or None without the latent map.
        self._boxes = ([None] * len(model_controls) if latent_bounds is None
                       else _box_bounds(latent_bounds, model_controls))

        # One scale per space: controls in the same space share it.
        scales = {}
        self._scales = []
        transformed = []
        for control, box in zip(model_controls, self._boxes):
            # The tape value, not ``control.control``: between stages the
            # controls move through ``Control.update``, which only changes it.
            model = control.tape_value()
            space = model.function_space()
            if space not in scales:
                scales[space] = _inverse_sqrt_lumped_mass(space)
            scale = scales[space]
            self._scales.append(scale)
            v = model.dat.data_ro
            if box is not None:
                lower, width = box
                free = width > 0.0
                t = np.full(width.shape, 0.5)
                t[free] = np.clip((v[free] - lower[free]) / width[free],
                                  1e-12, 1.0 - 1e-12)
                v = np.log(t) - np.log1p(-t)
            v_tilde = fire.Function(space)
            v_tilde.dat.data_wo[:] = v / scale.dat.data_ro
            transformed.append(Control(v_tilde, riesz_map="l2"))
        self._controls = Enlist(model_controls.delist(transformed))
        self._last = [control.control.copy(deepcopy=True)
                      for control in self._controls]

    @property
    def controls(self) -> Enlist:
        r""":class:`pyadjoint.enlisting.Enlist`: the controls over :math:`\tilde{v}`."""
        return self._controls

    def __call__(self, values):
        r"""Return :math:`\hat{J}(\tilde{v}) = J(m)`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\tilde{v}`, one per control.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value.
        """
        for last, value in zip(self._last, Enlist(values)):
            last.assign(value)
        return self._functional(self._model_controls.delist(self.map_result(values)))

    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        r"""Return :math:`D\hat{J}(\tilde{v}) = M_L^{-1/2}\, \frac{dm}{dv}\, DJ(m)`.

        :math:`dm/dv` is 1, or :math:`(u - \ell)\,\sigma(1 - \sigma)` with
        ``latent_bounds``, taken at the last :math:`\tilde{v}` the functional
        was evaluated at.

        Parameters
        ----------
        adj_input : float, optional
            Adjoint value of the functional result.
        apply_riesz : bool, optional
            Whether to return the gradient instead of the derivative.

        Returns
        -------
        firedrake.Cofunction, firedrake.Function or list
            One per control.
        """
        derivatives = []
        for value, last, scale, box, control in zip(
            Enlist(self._functional.derivative(adj_input=adj_input)),
            self._last, self._scales, self._boxes, self._controls,
        ):
            factor = scale.dat.data_ro
            if box is not None:
                sigma = _sigmoid(last.dat.data_ro * scale.dat.data_ro)
                factor = factor * box[1] * sigma * (1.0 - sigma)
            derivative = fire.Cofunction(scale.function_space().dual())
            derivative.dat.data_wo[:] = value.dat.data_ro * factor
            if apply_riesz:
                derivative = control.control._ad_convert_riesz(
                    derivative, riesz_map="l2",
                )
            derivatives.append(derivative)
        return self._controls.delist(derivatives)

    def tlm(self, m_dot):
        """Not provided: spyro's inversions use first derivatives only.

        Parameters
        ----------
        m_dot : firedrake.Function or sequence of firedrake.Function
            Direction, one per control.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "LumpedL2ReducedFunctional does not provide tangent linear "
            "actions: spyro's inversions run BQNLS, a quasi-Newton method "
            "that only needs gradients.",
        )

    def hessian(self, m_dot, hessian_input=None, evaluate_tlm: bool = True,
                apply_riesz: bool = False):
        """Not provided: spyro's inversions use first derivatives only.

        Parameters
        ----------
        m_dot : firedrake.Function or sequence of firedrake.Function
            Direction, one per control.
        hessian_input : pyadjoint.OverloadedType, optional
            Hessian value of the functional result.
        evaluate_tlm : bool, optional
            Whether to evaluate the tangent linear model first.
        apply_riesz : bool, optional
            Whether to return the primal result instead of the dual one.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "LumpedL2ReducedFunctional does not provide Hessian actions: "
            "spyro's inversions run BQNLS, a quasi-Newton method that only "
            "needs gradients.",
        )

    def map_result(self, values) -> list:
        r"""Return the model :math:`m` at :math:`\tilde{v}`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\tilde{v}`, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`m`, one per control, named after the model controls.
        """
        models = []
        for value, scale, box, control in zip(
            Enlist(values), self._scales, self._boxes, self._model_controls,
        ):
            v = value.dat.data_ro * scale.dat.data_ro
            if box is not None:
                lower, width = box
                v = lower + width * _sigmoid(v)
            model = fire.Function(scale.function_space(), name=control.control.name())
            model.dat.data_wo[:] = v
            models.append(model)
        return models

    def transform_bounds(self, bounds) -> list:
        r"""Return bounds on :math:`\tilde{v}` from bounds on :math:`m`.

        Parameters
        ----------
        bounds : sequence of tuple
            ``(lower, upper)`` on :math:`m`, one per control. Each bound is a
            scalar, a field, or None.

        Returns
        -------
        list of tuple
            ``(lower, upper)`` on :math:`\tilde{v}`, one per control.

        Raises
        ------
        ValueError
            With ``latent_bounds``: the latent controls need no bounds.
        """
        if self._boxes[0] is not None:
            raise ValueError(
                "With latent_bounds the model stays inside its bounds by "
                "construction, so the latent controls take no bounds.",
            )
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
