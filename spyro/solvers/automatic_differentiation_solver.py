from contextlib import contextmanager

import firedrake.adjoint as fire_adj
from pyadjoint import (
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)
from pyadjoint.tape import get_working_tape


class AutomatedAdjoint:
    """Automatic differentiation wrapper for seismic inversion."""

    def __init__(self, control, ensemble, model_control_index=0):
        if isinstance(control, (list, tuple)):
            controls = tuple(control)
        else:
            controls = (control,)
        if len(controls) == 0:
            raise ValueError("AutomatedAdjoint requires at least one control.")
        if not 0 <= model_control_index < len(controls):
            raise ValueError("model_control_index is out of range for controls.")

        self._controls = controls
        self._ensemble = ensemble
        self._model_control_index = model_control_index
        self._functional = None
        self._reduced_functional = None
        self._tape = None

    @property
    def control(self):
        """Primary model control associated with the reduced functional."""
        return self._controls[self._model_control_index]

    @property
    def controls(self):
        """All controls associated with the reduced functional."""
        return self._controls

    @property
    def reduced_functional(self):
        """Reduced functional object for the current controls."""
        return self._reduced_functional

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
        controls = [
            fire_adj.Control(control)
            for control in self._controls
        ]
        derivative_components = None
        if len(controls) > 1:
            derivative_components = (self._model_control_index,)
            reduced_controls = controls
        else:
            reduced_controls = controls[0]
        self._reduced_functional = fire_adj.EnsembleReducedFunctional(
            functional,
            reduced_controls,
            self._ensemble,
            derivative_components=derivative_components,
            tape=self._tape,
        )
        return self._reduced_functional

    def recompute_functional(self, control_value):
        """Recompute the functional at a control value."""
        return self._reduced_functional(control_value)

    def compute_gradient(self):
        """Compute the Riesz-represented gradient for the control."""
        return self._reduced_functional.derivative(apply_riesz=True)

    def compute_derivative(self):
        """Compute the derivative with respect to the control."""
        return self._reduced_functional.derivative(apply_riesz=False)

    def verify_gradient(self, control_value, direction=None, dJdm=None):
        """Run a Taylor test for the reduced functional."""
        if control_value is None:
            if len(self._controls) > 1:
                raise NotImplementedError(
                    "verify_gradient requires explicit control_value and "
                    "direction when multiple controls are used."
                )
            control_value = self.control

        if direction is None:
            direction = control_value.copy(deepcopy=True)
            direction.interpolate(1.)

        return fire_adj.taylor_test(
            self._reduced_functional,
            control_value,
            direction,
            dJdm=dJdm,
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
