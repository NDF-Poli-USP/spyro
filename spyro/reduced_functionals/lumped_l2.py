"""A reduced functional over controls rescaled by their lumped mass."""

import firedrake as fire
import numpy as np

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
        The scale taking :math:`z` to :math:`m = M_L^{-1/2} z`.

    Raises
    ------
    ValueError
        If the lumped mass is not positive, as for quadratic Lagrange on
        triangles and tetrahedra.
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
    scale = fire.Function(function_space)
    scale.dat.data_wo[:] = 1.0 / np.sqrt(mass.dat.data_ro)
    return scale


class LumpedL2ReducedFunctional(AbstractReducedFunctional):
    r"""A reduced functional over controls rescaled by their lumped mass.

    Represents :math:`\hat{J}(z) = J(M_L^{-1/2} z)`: each control :math:`m`
    of the wrapped functional is replaced by :math:`z = M_L^{1/2} m`, with
    :math:`M_L` its lumped mass.

    :math:`M_L` is the mass matrix assembled with the quadrature spyro adopts
    for each element (:func:`spyro.domains.quadrature.quadrature_rules`). For
    KMV elements and spectral (GLL) quadrilaterals the quadrature points are
    the nodes, so this matrix is diagonal at any degree.

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
