from contextlib import contextmanager

import firedrake.adjoint as fire_adj
from pyadjoint import (
    ReducedFunctional,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)
from pyadjoint.tape import get_working_tape


class AutomatedAdjoint:
    """Automatic differentiation wrapper for seismic inversion."""

    def __init__(self, control):
        self._control = control
        self._functional = None
        self._reduced_functional = None
        self._tape = None

    @property
    def control(self):
        """Control object associated with the reduced functional."""
        return self._control

    @property
    def reduced_functional(self):
        """Reduced functional object for the current control."""
        return self._require_reduced_functional()

    @contextmanager
    def fresh_tape(self):
        """Use a fresh working tape for one annotated evaluation."""
        self.clear_tape()
        pause_annotation()
        self._tape = Tape()
        with set_working_tape(self._tape):
            try:
                yield self
            finally:
                pause_annotation()

    def create_reduced_functional(self, functional):
        """Create the reduced functional after the recorded forward solve."""
        if self._tape is None:
            self._tape = get_working_tape()
        if self._tape is None:
            raise RuntimeError(
                "No adjoint tape is available. Use fresh_tape() or set a "
                "working tape before creating the reduced functional."
            )
        self._functional = functional
        self._reduced_functional = ReducedFunctional(
            functional,
            fire_adj.Control(self._control),
            tape=self._tape,
        )
        return self._reduced_functional

    def _require_reduced_functional(self):
        if self._reduced_functional is None:
            raise RuntimeError(
                "No reduced functional has been created. Call "
                "create_reduced_functional(functional) first."
            )
        return self._reduced_functional

    def recompute_functional(self, control_value):
        """Recompute the functional at a control value."""
        return self._require_reduced_functional()(control_value)

    def compute_gradient(self):
        """Compute the Riesz-represented gradient for the control."""
        return self._require_reduced_functional().derivative(apply_riesz=True)

    def compute_derivative(self):
        """Compute the derivative with respect to the control."""
        return self._require_reduced_functional().derivative(apply_riesz=False)

    def verify_gradient(self, control_value=None, direction=None):
        """Run a first-order Taylor test for the reduced functional."""
        reduced_functional = self._require_reduced_functional()
        if control_value is None:
            control_value = self._control
        if direction is None:
            direction = control_value.copy(deepcopy=True)
            direction.interpolate(1.0)
        return fire_adj.taylor_test(
            reduced_functional,
            control_value,
            direction,
        )

    def clear_tape(self):
        """Clear the stored tape and drop the reduced functional."""
        if self._tape is not None:
            self._tape.clear_tape()
        self._functional = None
        self._tape = None
        self._reduced_functional = None

    def stop_recording(self):
        """Stop recording operations on the adjoint tape."""
        pause_annotation()

    def start_recording(self):
        """Start recording operations on the adjoint tape."""
        if self._tape is None:
            self._tape = get_working_tape()
        if self._tape is None:
            raise RuntimeError(
                "No adjoint tape is available. Use fresh_tape() or set a "
                "working tape before starting annotation."
            )
        continue_annotation()
