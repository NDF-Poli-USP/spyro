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
        If that quadrature does not give a diagonal mass matrix.
    """
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
    largest = off_diagonal.norm(fire.PETSc.NormType.INFINITY) / diagonal.max()[1]
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

    inverse_sqrt_mass = fire.Function(function_space)
    with inverse_sqrt_mass.dat.vec_wo as values:
        diagonal.copy(values)
        values.sqrtabs()
        values.reciprocal()
    return inverse_sqrt_mass


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

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional of the model controls.

    Raises
    ------
    ValueError
        If the quadrature spyro adopts for a control space does not give a
        diagonal mass matrix.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional):
        super().__init__()
        self._functional = reduced_functional
        model_controls = reduced_functional.controls
        if not isinstance(model_controls, Enlist):
            model_controls = Enlist(model_controls)
        self._model_controls = model_controls

        # One M_L^{-1/2} per space: controls in the same space share it.
        inverse_sqrt_masses = {}
        self._inverse_sqrt_masses = []
        m_tilde_controls = []
        for control in model_controls:
            # The tape value, not ``control.control``: between stages the
            # controls move through ``Control.update``, which only changes it.
            model = control.tape_value()
            space = model.function_space()
            if space not in inverse_sqrt_masses:
                inverse_sqrt_masses[space] = _inverse_sqrt_lumped_mass(space)
            inverse_sqrt_mass = inverse_sqrt_masses[space]
            self._inverse_sqrt_masses.append(inverse_sqrt_mass)
            m_tilde = fire.Function(space)
            m_tilde.dat.data_wo[:] = model.dat.data_ro / inverse_sqrt_mass.dat.data_ro
            m_tilde_controls.append(Control(m_tilde, riesz_map="l2"))
        self._controls = Enlist(model_controls.delist(m_tilde_controls))

    @property
    def controls(self) -> Enlist:
        r""":class:`pyadjoint.enlisting.Enlist`: the controls :math:`\tilde{m} = M_L^{1/2} m`.

        Each is a model control :math:`m` scaled by the square root of its
        lumped mass, so the Euclidean inner product of :math:`\tilde{m}` is
        the lumped :math:`L^2` inner product of :math:`m`.
        """
        return self._controls

    def _divide_by_sqrt_mass(self, values, dual: bool) -> list:
        r"""Divide each value by the square root of its control's lumped mass.

        That is :math:`M_L^{-1/2}` times each value: it takes the controls
        :math:`\tilde{m}` to the model :math:`m`, and the derivative
        :math:`DJ(m)` to the derivative in :math:`\tilde{m}`.

        Parameters
        ----------
        values : firedrake.Function, firedrake.Cofunction or sequence
            One per control.
        dual : bool
            Whether the values are Cofunctions.

        Returns
        -------
        list
            The divided values, one per control.
        """
        divided = []
        for value, inverse_sqrt_mass in zip(Enlist(values), self._inverse_sqrt_masses):
            space = inverse_sqrt_mass.function_space()
            out = fire.Cofunction(space.dual()) if dual else fire.Function(space)
            out.dat.data_wo[:] = value.dat.data_ro * inverse_sqrt_mass.dat.data_ro
            divided.append(out)
        return divided

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
        models = self._divide_by_sqrt_mass(values, dual=False)
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
        derivatives = self._divide_by_sqrt_mass(derivative, dual=True)
        if apply_riesz:
            derivatives = [
                control.control._ad_convert_riesz(value, riesz_map="l2")
                for control, value in zip(self._controls, derivatives)
            ]
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
        r"""Return the model controls :math:`m = M_L^{-1/2} \tilde{m}`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\tilde{m}`, the model controls scaled by the square root
            of their lumped mass, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`m`, one per control, named after the model controls.
        """
        models = self._divide_by_sqrt_mass(values, dual=False)
        for model, control in zip(models, self._model_controls):
            model.rename(control.control.name())
        return models
