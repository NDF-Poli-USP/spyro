import firedrake.adjoint as fire_adj
import firedrake as fire


class AutomaticDifferentiationSolver:
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
        self._reduced_functional = fire_adj.ReducedFunctional(
            functional, fire_adj.Control(self._control))

    def compute_gradient(self, control_value=None, apply_riesz=True):
        """Compute the gradient with respect to the control.

        Parameters
        ----------
        control_value : firedrake.Function, optional
            Control value where the reduced functional is evaluated
            before taking the derivative.
        apply_riesz : bool, optional
            If True, return the primal gradient (Riesz representer).

        Returns
        -------
        gradient : firedrake.Function
            Gradient of the objective function with respect to the control.
        """
        if control_value is not None:
            self._reduced_functional(control_value)
        self.gradient = self._reduced_functional.derivative(
            apply_riesz=apply_riesz)
        return self.gradient

    def compute_derivative(self, control_value=None, apply_riesz=False):
        return self.compute_gradient(
            control_value=control_value, apply_riesz=apply_riesz)

    @property
    def reduced_functional(self):
        """Reduced functional object associated with the current control."""
        return self._reduced_functional

    @property
    def control(self):
        """Control object associated with the reduced functional."""
        return self._control

    def verify_gradient(self, control_value=None, direction=None):
        """Run a first-order Taylor test for the reduced functional.

        Parameters
        ----------
        control_value : firedrake.Function, optional
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
        return fire_adj.taylor_test(
            self._reduced_functional, control_value, direction)
