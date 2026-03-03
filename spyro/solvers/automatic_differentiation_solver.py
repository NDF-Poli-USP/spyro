import firedrake.adjoint as fire_adj
import firedrake as fire
from pyadjoint import ReducedFunctional


class SpyroReducedFunctional(ReducedFunctional):
    """Automatic differentiation solver for seismic inversion.

    This class provides an interface to compute the gradient of a functional
    with respect to a control variable using the automatic differentiation
    capabilities of Firedrake.

    Parameters
    ----------
    functional : firedrake.AdjFloat
        The functional for which the gradient is to be computed.
    control : firedrake.Function
        The control variable with respect to which the gradient is computed.
    """

    def __init__(self, functional, control):
        self.gradient = None
        self._control = control
        super().__init__(functional, fire_adj.Control(control))

    def compute_gradient(self):
        """Compute the gradient with respect to the control.

        Parameters
        ----------
        apply_riesz : bool, optional
            If True, return the primal gradient (Riesz representer).

        Returns
        -------
        gradient : firedrake.Function
            Gradient of the objective function with respect to the control.
        """
        return self.derivative(apply_riesz=True)

    def compute_derivative(self):
        """Compute the derivative with respect to the control.
        """
        return self.derivative(apply_riesz=False)

    @property
    def control(self):
        """Control object associated with the reduced functional."""
        return self._control

    def verify_gradient(self, control_value=None, direction=None):
        """Run a first-order Taylor test for the reduced functional.

        Parameters
        ----------
        control_value : firedrake.Function or list of firedrake.Functions, optional
            Expansion point for the Taylor test.
        direction : firedrake.Function, optional
            Perturbation direction used by the Taylor test.

        Returns
        -------
        float
            Smallest measured convergence rate from the Taylor test.
        """
        if control_value is None:
            control_value = self._control
        if direction is None:
            direction = control_value.copy(deepcopy=True)
            direction.interpolate(1.0)
        return fire_adj.taylor_test(self, control_value, direction)
