"""A reduced functional over latent controls mapped into their bounds."""

import firedrake as fire
import numpy as np

from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional


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
        raise ValueError("LatentReducedFunctional needs a lower and an "
                         "upper bound for every control.")
    if isinstance(bound, fire.Function):
        return bound.dat.data_ro.copy()
    return np.full(fire.Function(space).dat.data_ro.shape, float(bound))


class LatentReducedFunctional(AbstractReducedFunctional):
    r"""A reduced functional over latent controls :math:`\psi`.

    Represents :math:`\hat{J}(\psi) = J(m(\psi))`, with

    .. math::

        m = \ell + (u - \ell)\,\sigma(\psi), \qquad
        \sigma(\psi) = \frac{1}{1 + e^{-\psi}},

    applied at each degree of freedom. Any :math:`\psi` gives a model
    strictly inside the bounds :math:`[\ell, u]`, so the optimization over
    :math:`\psi` needs no bounds. Degrees of freedom with :math:`\ell = u`
    stay fixed.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional of the model controls.
    bounds : sequence of tuple
        ``(lower, upper)`` on :math:`m`, one per control. Each bound is a
        scalar or a field.

    Raises
    ------
    ValueError
        If a bound is None.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional, bounds):
        super().__init__()
        self._functional = reduced_functional
        model_controls = reduced_functional.controls
        if not isinstance(model_controls, Enlist):
            model_controls = Enlist(model_controls)
        self._model_controls = model_controls

        self._lower = []
        self._width = []
        latent = []
        for control, (lower, upper) in zip(model_controls, bounds):
            # The tape value: Control.update moves it, not the Function.
            model = control.tape_value()
            space = model.function_space()
            lower = _bound_array(lower, space)
            width = _bound_array(upper, space) - lower
            self._lower.append(lower)
            self._width.append(width)
            free = width > 0.0
            t = np.full(width.shape, 0.5)
            t[free] = np.clip((model.dat.data_ro[free] - lower[free])
                              / width[free], 1e-12, 1.0 - 1e-12)
            psi = fire.Function(space)
            psi.dat.data_wo[:] = np.log(t) - np.log1p(-t)
            latent.append(Control(psi))
        self._controls = Enlist(model_controls.delist(latent))
        self._last = [control.control.copy(deepcopy=True)
                      for control in self._controls]

    @property
    def controls(self) -> Enlist:
        r""":class:`pyadjoint.enlisting.Enlist`: the controls over :math:`\psi`."""
        return self._controls

    def map_result(self, values) -> list:
        r"""Return :math:`m(\psi)`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\psi`, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`m`, one per control.
        """
        models = []
        for psi, lower, width, control in zip(
            Enlist(values), self._lower, self._width, self._model_controls,
        ):
            model = fire.Function(psi.function_space(),
                                  name=control.control.name())
            model.dat.data_wo[:] = lower + width * _sigmoid(psi.dat.data_ro)
            models.append(model)
        return models

    def __call__(self, values):
        r"""Return :math:`\hat{J}(\psi) = J(m(\psi))`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`\psi`, one per control.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value.
        """
        for last, psi in zip(self._last, Enlist(values)):
            last.assign(psi)
        models = self.map_result(values)
        return self._functional(self._model_controls.delist(models))

    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        r"""Return :math:`D\hat{J}(\psi) = DJ(m)\,(u - \ell)\,\sigma(1 - \sigma)`.

        Taken at the last :math:`\psi` the functional was evaluated at.

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
        for value, psi, width, control in zip(
            Enlist(self._functional.derivative(adj_input=adj_input)),
            self._last, self._width, self._controls,
        ):
            sigma = _sigmoid(psi.dat.data_ro)
            derivative = fire.Cofunction(psi.function_space().dual())
            derivative.dat.data_wo[:] = (
                value.dat.data_ro * width * sigma * (1.0 - sigma)
            )
            if apply_riesz:
                derivative = control.control._ad_convert_riesz(
                    derivative, riesz_map=control.riesz_map,
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
            "LatentReducedFunctional does not provide tangent linear "
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
            "LatentReducedFunctional does not provide Hessian actions: "
            "spyro's inversions run BQNLS, a quasi-Newton method that only "
            "needs gradients.",
        )
