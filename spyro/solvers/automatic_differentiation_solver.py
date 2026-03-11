import firedrake.adjoint as fire_adj
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

    def recompute_functional(self, control_value):
        """Recompute the forward solution to reduce memory usage.

        Parameters
        ----------
        control_value : firedrake.Function
            The control value at which to recompute the functional.

        Returns
        -------
        float
            The value of the functional at the given control value.
        """
        return self.functional(control_value)

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

    def clear_tape(self):
        """Clear the adjoint tape to free memory."""
        fire_adj.get_working_tape().clear()

    def stop_recording(self):
        """Stop recording operations on the adjoint tape."""
        fire_adj.stop_annotation()

    def start_recording(self):
        """Start recording operations on the adjoint tape."""
        fire_adj.start_annotation()
