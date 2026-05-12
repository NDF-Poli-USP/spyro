from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad


class AutomatedAdjoint:
    def __init__(self, controls=None):
        self.controls = controls
        self.reduced_functional = None
        self._tape = None

    @contextmanager
    def fresh_tape(self):
        self.clear_tape()
        self._tape = Tape()
        fire_ad.set_working_tape(self._tape)
        continue_annotation()
        try:
            yield self._tape
        finally:
            pause_annotation()

    def start_recording(self):
        if self._tape is None:
            self._tape = Tape()
            fire_ad.set_working_tape(self._tape)
        continue_annotation()
        return self._tape

    def stop_recording(self):
        pause_annotation()

    def clear_tape(self):
        self.reduced_functional = None
        self._tape = None
        fire_ad.set_working_tape(Tape())
        pause_annotation()

    def create_reduced_functional(self, functional):
        control = fire_ad.Control(self.controls)
        self.reduced_functional = fire_ad.ReducedFunctional(
            functional,
            control,
            tape=self._tape,
        )
        return self.reduced_functional

    def recompute_functional(self, control_value):
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional(control_value)

    def compute_gradient(self):
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=True)

    def compute_derivative(self):
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=False)

    def verify_gradient(self, control_var, direction=None, dJdm=None):
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        if direction is None:
            direction = fire.Function(control_var.function_space())
            direction.interpolate(0.01)
        # pyadjoint's ``taylor_test`` expects ``dJdm`` to be the scalar
        # directional derivative ``J'(m)(h)``, not the gradient itself. When a
        # Firedrake ``Function`` (Riesz representer of the gradient) or a
        # ``Cofunction`` (raw derivative) is supplied, reduce it to a scalar by
        # pairing it with the perturbation ``direction``. Otherwise ``eps *
        # dJdm`` inside pyadjoint becomes a UFL expression and the comparison
        # ``min(residuals) < 1E-15`` raises ``UFL conditions cannot be
        # evaluated as bool in a Python context``.
        if dJdm is not None and not isinstance(dJdm, (int, float)):
            if isinstance(dJdm, fire.Function):
                dJdm = fire.assemble(
                    fire.inner(dJdm, direction) * fire.dx
                )
            elif isinstance(dJdm, fire.Cofunction):
                # Apply the cofunction to the direction (duality pairing).
                dJdm = fire.assemble(fire.action(dJdm, direction))
            else:
                # Unknown type, fall back to pyadjoint's internal computation.
                dJdm = None
        return taylor_test(self.reduced_functional, control_var, direction, dJdm=dJdm)
