"""A reduced functional over controls rescaled by their lumped mass."""

import firedrake as fire

from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional

from ..domains.quadrature import quadrature_rules


@no_annotations
def _inverse_sqrt_lumped_mass(
    function_space: fire.FunctionSpace,
) -> fire.Function:
    r"""Return :math:`M_L^{-1/2}`, the inverse square root of the lumped mass.

    See :class:`LumpedL2ReducedFunctional` for the quadrature it is assembled
    with.

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

    scale = fire.Function(function_space)
    with scale.dat.vec_wo as values:
        diagonal.copy(values)
        values.sqrtabs()
        values.reciprocal()
    return scale


class LumpedL2ReducedFunctional(AbstractReducedFunctional):
    r"""A reduced functional over controls rescaled by their lumped mass.

    Represents :math:`\hat{J}(\tilde{m}) = J(M_L^{-1/2} \tilde{m})`: each control :math:`m`
    of the wrapped functional is replaced by :math:`\tilde{m} = M_L^{1/2} m`, with
    :math:`M_L` its lumped mass.

    :math:`M_L` is the mass matrix assembled with the quadrature spyro adopts
    for each element (:func:`spyro.domains.quadrature.quadrature_rules`). For
    KMV elements and spectral (GLL) quadrilaterals the quadrature points are
    the nodes, so this matrix is diagonal at any degree. Control spaces where
    it is not diagonal are rejected.

    The controls here carry the ``"l2"`` Riesz map, which in :math:`\tilde{m}` is the
    lumped :math:`L^2` one in :math:`m`. Values go in and come out as
    :math:`\tilde{m}`; :meth:`map_result` takes them back to :math:`m`, and
    :meth:`transform_bounds` takes bounds on :math:`m` to bounds on
    :math:`\tilde{m}`.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional of the model controls.

    Raises
    ------
    ValueError
        If the mass matrix of a control space is not diagonal.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional):
        super().__init__()
        self._functional = reduced_functional
        model_controls = reduced_functional.controls
        if not isinstance(model_controls, Enlist):
            model_controls = Enlist(model_controls)
        self._model_controls = model_controls

        # One scale per space: controls in the same space share it.
        scales = {}
        self._scales = []
        transformed = []
        for control in model_controls:
            model = control.control
            space = model.function_space()
            if space not in scales:
                scales[space] = _inverse_sqrt_lumped_mass(space)
            scale = scales[space]
            self._scales.append(scale)
            m_tilde = fire.Function(space)
            m_tilde.dat.data_wo[:] = model.dat.data_ro / scale.dat.data_ro
            transformed.append(Control(m_tilde, riesz_map="l2"))
        self._controls = Enlist(model_controls.delist(transformed))

    @property
    def controls(self) -> Enlist:
        r""":class:`pyadjoint.enlisting.Enlist`: the controls over :math:`\tilde{m}`."""
        return self._controls

    def _scaled(self, values, dual: bool) -> list:
        """Return :math:`M_L^{-1/2}` times each value.

        Parameters
        ----------
        values : firedrake.Function, firedrake.Cofunction or sequence
            One per control.
        dual : bool
            Whether the values are Cofunctions.

        Returns
        -------
        list
            The scaled values, one per control.
        """
        scaled = []
        for value, scale in zip(Enlist(values), self._scales):
            space = scale.function_space()
            out = fire.Cofunction(space.dual()) if dual else fire.Function(space)
            out.dat.data_wo[:] = value.dat.data_ro * scale.dat.data_ro
            scaled.append(out)
        return scaled

    def __call__(self, values):
        r"""Return :math:`\hat{J}(\tilde{m}) = J(M_L^{-1/2} \tilde{m})`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\tilde{m}`, one per control.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value.
        """
        models = self._scaled(values, dual=False)
        return self._functional(self._model_controls.delist(models))

    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        r"""Return :math:`D\hat{J}(\tilde{m}) = M_L^{-1/2} DJ(m)`.

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
        derivative = self._functional.derivative(adj_input=adj_input)
        scaled = self._scaled(derivative, dual=True)
        if apply_riesz:
            scaled = [
                control.control._ad_convert_riesz(value, riesz_map="l2")
                for control, value in zip(self._controls, scaled)
            ]
        return self._controls.delist(scaled)

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
        r"""Return :math:`m = M_L^{-1/2} \tilde{m}`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\tilde{m}`, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`m`, one per control.
        """
        models = self._scaled(values, dual=False)
        for model, control in zip(models, self._model_controls):
            model.rename(control.control.name())
        return models

    def transform_bounds(self, bounds) -> list:
        r"""Return bounds on :math:`\tilde{m}` from bounds on :math:`m`.

        Parameters
        ----------
        bounds : sequence of tuple
            ``(lower, upper)`` on :math:`m`, one per control. Each bound is a
            scalar, a field, or None.

        Returns
        -------
        list of tuple
            ``(lower, upper)`` on :math:`\tilde{m}`, one per control.
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
