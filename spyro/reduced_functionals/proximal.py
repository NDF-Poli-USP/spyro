"""A reduced functional with a proximal term around an anchor."""

import firedrake as fire
import numpy as np

from pyadjoint import no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional

from .latent import _bound_array
from .lumped_l2 import _lumped_mass

PROXIMAL_KINDS = ("l2", "bregman")


class ProximalReducedFunctional(AbstractReducedFunctional):
    r"""A reduced functional plus a proximal term around an anchor :math:`a`.

    Represents

    .. math::

        J(v) + \frac{1}{\alpha} \sum_c s_c\, D(v_c, a_c),

    with :math:`D` integrated with the lumped mass :math:`M_L`, so
    :math:`D(v, a) = \sum_i M_{L,ii}\, d(v_i, a_i)`. Two choices of :math:`d`:

    ``"l2"``
        :math:`d(v, a) = \frac12 (v - a)^2`.
    ``"bregman"``
        The Bregman divergence of the entropy of the box :math:`[\ell, u]`,

        .. math::

            d(v, a) = w \left[t \ln\frac{t}{t_a}
                      + (1 - t) \ln\frac{1 - t}{1 - t_a}\right],
            \qquad t = \frac{v - \ell}{w},\ w = u - \ell,

        whose derivative is :math:`\operatorname{logit}(t) -
        \operatorname{logit}(t_a)`. It grows without bound at the box, so
        :math:`v` has to stay strictly inside it.

    Parameters
    ----------
    reduced_functional : pyadjoint.reduced_functional.AbstractReducedFunctional
        Functional :math:`J` of the controls :math:`v`.
    kind : str
        ``"l2"`` or ``"bregman"``.
    step : float, optional
        :math:`\alpha`; the proximal term is weighted by :math:`1/\alpha`.
    scales : sequence of float, optional
        :math:`s_c`, one per control. Default 1.
    bounds : sequence of tuple, optional
        ``(lower, upper)`` on :math:`v`, one per control, each a scalar or a
        field. Required for ``"bregman"``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or ``"bregman"`` is given no bounds.
    """

    @no_annotations
    def __init__(self, reduced_functional: AbstractReducedFunctional,
                 kind: str, step: float = 1.0, scales=None, bounds=None):
        super().__init__()
        if kind not in PROXIMAL_KINDS:
            raise ValueError(f"kind must be one of {PROXIMAL_KINDS}, "
                             f"not '{kind}'.")
        if kind == "bregman" and bounds is None:
            raise ValueError("The 'bregman' proximal term needs bounds.")
        self._functional = reduced_functional
        controls = reduced_functional.controls
        if not isinstance(controls, Enlist):
            controls = Enlist(controls)
        self._controls = controls
        self._kind = kind
        self._step = float(step)
        self._scales = [1.0] * len(controls) if scales is None else list(scales)

        masses = {}
        self._masses = []
        self._anchors = []
        self._lower = []
        self._width = []
        for index, control in enumerate(controls):
            # The tape value: Control.update moves it, not the Function.
            value = control.tape_value()
            space = value.function_space()
            if space not in masses:
                masses[space] = _lumped_mass(space).dat.data_ro.copy()
            self._masses.append(masses[space])
            self._anchors.append(value.dat.data_ro.copy())
            if kind == "bregman":
                lower, upper = bounds[index]
                lower = _bound_array(lower, space)
                self._lower.append(lower)
                self._width.append(_bound_array(upper, space) - lower)
        self._last = [control.tape_value().copy(deepcopy=True)
                      for control in controls]

    @property
    def controls(self) -> Enlist:
        """:class:`pyadjoint.enlisting.Enlist`: the controls of the wrapped functional."""
        return self._controls

    def update_anchor(self, values) -> None:
        """Move the anchor :math:`a` to ``values``.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            New anchor, one per control.
        """
        self._anchors = [value.dat.data_ro.copy() for value in Enlist(values)]

    def _unit(self, index: int, values: np.ndarray) -> np.ndarray:
        """Return :math:`t = (v - \\ell)/w`, kept inside :math:`(0, 1)`.

        Parameters
        ----------
        index : int
            Control index.
        values : numpy.ndarray
            :math:`v` at the degrees of freedom.

        Returns
        -------
        numpy.ndarray
            :math:`t`, with degrees of freedom where :math:`w = 0` at 1/2.
        """
        width = self._width[index]
        t = np.full(values.shape, 0.5)
        free = width > 0.0
        t[free] = np.clip((values[free] - self._lower[index][free])
                          / width[free], 1e-12, 1.0 - 1e-12)
        return t

    def proximal_value(self, values) -> float:
        r"""Return the proximal term :math:`\frac{1}{\alpha}\sum_c s_c D(v_c, a_c)`.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`v`, one per control.

        Returns
        -------
        float
            The term, summed over all processes.
        """
        total = 0.0
        for index, value in enumerate(Enlist(values)):
            v = value.dat.data_ro
            anchor = self._anchors[index]
            if self._kind == "l2":
                density = 0.5 * (v - anchor) ** 2
            else:
                t = self._unit(index, v)
                t_a = self._unit(index, anchor)
                density = self._width[index] * (
                    t * np.log(t / t_a)
                    + (1.0 - t) * np.log((1.0 - t) / (1.0 - t_a))
                )
            term = fire.Function(value.function_space())
            term.dat.data_wo[:] = self._scales[index] * self._masses[index] * density
            with term.dat.vec_ro as entries:
                total += entries.sum()
        return total / self._step

    def map_result(self, values) -> list:
        """Return copies of ``values``: the controls are those of the wrapped functional.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`v`, one per control.

        Returns
        -------
        list of firedrake.Function
            :math:`v`, one per control.
        """
        return [value.copy(deepcopy=True) for value in Enlist(values)]

    def __call__(self, values):
        r"""Return :math:`J(v)` plus the proximal term.

        Parameters
        ----------
        values : firedrake.Function or sequence of firedrake.Function
            :math:`v`, one per control.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value.
        """
        for last, value in zip(self._last, Enlist(values)):
            last.assign(value)
        return self._functional(values) + self.proximal_value(values)

    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        """Return the derivative of :math:`J` plus that of the proximal term.

        Taken at the last :math:`v` the functional was evaluated at.

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
        for index, (value, last, control) in enumerate(zip(
            Enlist(self._functional.derivative(adj_input=adj_input)),
            self._last, self._controls,
        )):
            v = last.dat.data_ro
            anchor = self._anchors[index]
            if self._kind == "l2":
                slope = v - anchor
            else:
                t = self._unit(index, v)
                t_a = self._unit(index, anchor)
                slope = (np.log(t) - np.log1p(-t)) - (np.log(t_a) - np.log1p(-t_a))
                slope[self._width[index] <= 0.0] = 0.0
            derivative = value.copy(deepcopy=True)
            derivative.dat.data_wo[:] = value.dat.data_ro + (
                self._scales[index] * self._masses[index] * slope / self._step
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
            "ProximalReducedFunctional does not provide tangent linear "
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
            "ProximalReducedFunctional does not provide Hessian actions: "
            "spyro's inversions run BQNLS, a quasi-Newton method that only "
            "needs gradients.",
        )
