from contextlib import contextmanager
from math import log

import firedrake.adjoint as fire_adj
from pyadjoint import (
    ReducedFunctional,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)
from pyadjoint.tape import get_working_tape, stop_annotating


class AutomatedAdjoint:
    """Automatic differentiation wrapper for seismic inversion."""

    def __init__(self, control, model_control_index=0):
        if isinstance(control, (list, tuple)):
            controls = tuple(control)
        else:
            controls = (control,)
        if len(controls) == 0:
            raise ValueError("AutomatedAdjoint requires at least one control.")
        if not 0 <= model_control_index < len(controls):
            raise ValueError("model_control_index is out of range for controls.")

        self._controls = controls
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
        self._reduced_functional = ReducedFunctional(
            functional,
            reduced_controls,
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
        """Run a first-order Taylor test for the reduced functional."""
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

        with stop_annotating():
            Jm = self._reduced_functional(control_value)
            if dJdm is None:
                dJdm = direction._ad_dot(self._reduced_functional.derivative())

            print("Running Taylor test")
            epsilons = [0.01 / 2 ** i for i in range(4)]
            residuals = []
            for eps in epsilons:
                perturbed_control = control_value._ad_add(
                    direction._ad_mul(eps)
                )
                Jp = self._reduced_functional(perturbed_control)
                residuals.append(float(abs(Jp - Jm - eps * dJdm)))

            print(f"Computed residuals: {residuals}")
            rates = [
                log(residuals[i] / residuals[i - 1])
                / log(epsilons[i] / epsilons[i - 1])
                for i in range(1, len(epsilons))
            ]
            print(f"Computed convergence rates: {rates}")

        cutoff = len(residuals)
        while cutoff > 1 and residuals[cutoff - 1] >= residuals[cutoff - 2]:
            cutoff -= 1

        usable_rate_count = max(1, cutoff - 1)
        return min(rates[:usable_rate_count])

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
