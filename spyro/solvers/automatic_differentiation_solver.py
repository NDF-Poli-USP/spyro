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

    def verify_gradient(self, variable, direction=None):
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        if direction is None:
            direction = fire.Function(variable.function_space())
            direction.dat.data[:] = 1.0
        return taylor_test(self.reduced_functional, variable, direction)
