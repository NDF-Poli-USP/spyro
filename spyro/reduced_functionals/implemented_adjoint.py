"""A reduced functional differentiated by spyro's implemented adjoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import firedrake as fire

from pyadjoint import AdjFloat, Control, no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional

from ..utils.typing import AdjointType, WaveType

if TYPE_CHECKING:
    from ..solvers.inversion import FullWaveformInversion


class ImplementedAdjointReducedFunctional(AbstractReducedFunctional):
    r"""The FWI objective, differentiated by the implemented adjoint.

    Represents :math:`\hat{J}(m) = J(u(m))` for the velocity model :math:`m`
    of an acoustic inversion. Nothing is recorded on a pyadjoint tape: every
    evaluation re-runs the forward solve from the inversion driver, and every
    derivative re-runs the backward propagator over that forward solve. This
    is the same work the scipy path of
    :meth:`~spyro.solvers.inversion.FullWaveformInversion.run_fwi` does, now
    behind the interface every pyadjoint optimizer and wrapper understands.

    The implemented adjoint returns the :math:`L^2` gradient :math:`g` with
    respect to the mass matrix :math:`M` assembled with the quadrature spyro
    adopts for the element, which is diagonal for KMV elements. The
    derivative is recovered from it as :math:`DJ(m) = M g`, and the control
    carries that same inner product as its Riesz map, so ``apply_riesz=True``
    gives :math:`g` back.

    Parameters
    ----------
    fwi : FullWaveformInversion
        Driver of an acoustic inversion whose guess control is configured.

    Raises
    ------
    NotImplementedError
        If the inversion is not acoustic: the implemented adjoint is only
        written for acoustic media.
    ValueError
        If the inversion is set to use the automated adjoint, or has no
        guess control configured.

    Notes
    -----
    Evaluating and differentiating go through the driver's
    :meth:`~spyro.solvers.inversion.FullWaveformInversion.get_functional`
    and :meth:`~spyro.solvers.inversion.FullWaveformInversion.get_gradient`,
    so the functional history, the iteration counter, the files those write
    and the deprecated gradient mask behave as they do under ``run_fwi``.
    """

    @no_annotations
    def __init__(self, fwi: FullWaveformInversion):
        super().__init__()
        if fwi.wave_type is not WaveType.ISOTROPIC_ACOUSTIC:
            raise NotImplementedError(
                f"{fwi.wave_type.name} media have no implemented adjoint; "
                "differentiate them with the automated adjoint instead.",
            )
        if fwi.adjoint_type is AdjointType.AUTOMATED_ADJOINT:
            raise ValueError(
                "This inversion is set to use the automated adjoint, whose "
                "reduced functional is the recorded one; ask the driver for "
                "it with FullWaveformInversion.reduced_functional().",
            )
        control = fwi.control_parameters
        if control is None:
            raise ValueError("No guess control parameter has been configured.")

        self._fwi = fwi
        space = control.function_space()
        # The measure the backward propagator solves its gradient with, so
        # that M g reproduces the derivative it came from.
        self._measure = fire.dx(**fwi.wave.quadrature_rule)
        self._test = fire.TestFunction(space)
        riesz_map = fire.RieszMap(
            fire.inner(fire.TrialFunction(space), self._test) * self._measure,
        )
        self._controls = Enlist(Control(control, riesz_map=riesz_map))
        # Whether the driver holds a forward solve at the last value this
        # functional was called with, which is where derivatives are taken.
        self._evaluated = False

    @property
    def controls(self) -> Enlist:
        """:class:`pyadjoint.enlisting.Enlist`: the velocity model control.

        Its Riesz map is the :math:`L^2` inner product the implemented
        adjoint computes its gradient in.
        """
        return self._controls

    @no_annotations
    def __call__(self, values) -> AdjFloat:
        """Return the functional at a new velocity model.

        Parameters
        ----------
        values : firedrake.Function or sequence of one firedrake.Function
            The velocity model to evaluate at.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value, summed over every shot.

        Raises
        ------
        ValueError
            If more than one value is given.
        """
        values = Enlist(values)
        if len(values) != 1:
            raise ValueError(
                "An acoustic inversion has one control, its velocity model; "
                f"received {len(values)} values.",
            )
        (value,) = values
        self._controls[0].update(value)
        self._fwi.set_guess_control(value)
        functional = self._fwi.get_functional()
        self._evaluated = True
        return AdjFloat(functional)

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz: bool = False):
        """Return the derivative at the last velocity model evaluated.

        Before any evaluation, the functional is first evaluated at the
        control the driver was configured with.

        Parameters
        ----------
        adj_input : float, optional
            Adjoint value of the functional result, which scales the
            derivative.
        apply_riesz : bool, optional
            Whether to return the :math:`L^2` gradient instead of the
            derivative.

        Returns
        -------
        firedrake.Cofunction or firedrake.Function
            The derivative, or with ``apply_riesz`` the gradient.
        """
        if not self._evaluated:
            self(self._controls[0].tape_value())
        gradient = self._fwi.get_gradient(save=False, calculate_functional=False)
        scaled = fire.Function(gradient.function_space()).assign(
            float(adj_input) * gradient,
        )
        if apply_riesz:
            return self._controls.delist([scaled])
        derivative = fire.assemble(fire.inner(scaled, self._test) * self._measure)
        return self._controls.delist([derivative])

    def tlm(self, m_dot):
        """Not provided: the implemented adjoint has no tangent linear model.

        Parameters
        ----------
        m_dot : firedrake.Function
            Direction.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "The implemented adjoint provides first derivatives only; use "
            "the automated adjoint for tangent linear actions.",
        )

    def hessian(self, m_dot, hessian_input=None, evaluate_tlm: bool = True,
                apply_riesz: bool = False):
        """Not provided: the implemented adjoint has no second-order adjoint.

        Parameters
        ----------
        m_dot : firedrake.Function
            Direction.
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
            "The implemented adjoint provides first derivatives only; use "
            "the automated adjoint for Hessian actions.",
        )
