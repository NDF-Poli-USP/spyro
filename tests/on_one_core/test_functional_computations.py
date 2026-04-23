from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.wave import Wave
from spyro.utils.typing import FunctionalEvaluationMode


class DummyWave(Wave):
    """A dummy wave class for testing the wave propagator API without
    needing to set up a full wave solver.
    """
    def matrix_building(self):
        pass

    def _initialize_model_parameters(self):
        pass

    def _create_function_space(self):
        pass

    def _set_vstate(self, vstate):
        self._vstate = vstate

    def _get_vstate(self):
        return getattr(self, "_vstate", None)

    def _set_prev_vstate(self, vstate):
        self._prev_vstate = vstate

    def _get_prev_vstate(self):
        return getattr(self, "_prev_vstate", None)

    def _set_next_vstate(self, vstate):
        self._next_vstate = vstate

    def _get_next_vstate(self):
        return getattr(self, "_next_vstate", None)

    def get_receivers_output(self):
        return None

    def get_function(self):
        return None

    def get_function_name(self):
        return "dummy"

    def rhs_no_pml(self):
        return None


def _build_wave():
    wave = DummyWave.__new__(DummyWave)
    wave._parallelism_type = "custom"
    wave.number_of_sources = 1
    wave.shot_ids_per_propagation = [[0]]
    wave.sources = None
    wave.comm = SimpleNamespace(
        comm=SimpleNamespace(rank=0, size=1, barrier=lambda: None),
        ensemble_comm=SimpleNamespace(rank=0, size=1),
    )
    wave._dt = 0.1
    wave._final_time = 0.2
    return wave


def test_compute_functional_mode_defaults_to_none():
    wave = _build_wave()
    assert wave.functional_evaluation_mode is None


def test_enable_compute_functional_defaults_to_per_timestep():
    wave = _build_wave()

    wave.enable_compute_functional()
    assert wave.functional_evaluation_mode == FunctionalEvaluationMode.PER_TIMESTEP


def test_compute_functional_accepts_after_solve_mode():
    wave = _build_wave()

    wave.functional_evaluation_mode = FunctionalEvaluationMode.AFTER_SOLVE
    assert wave.functional_evaluation_mode == FunctionalEvaluationMode.AFTER_SOLVE


def _base_functional_dictionary():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "method": "MLT",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.25, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": [(-0.2, 0.3), (-0.2, 0.5), (-0.2, 0.7)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.05,
            "dt": 0.001,
            "amplitude": 1,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": "results/forward_output.pvd",
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": "results/gradient.pvd",
            "adjoint_output": False,
            "adjoint_filename": None,
            "debug_output": False,
        },
    }


def _build_acoustic_wave(model_dictionary, velocity, *, real_shot_record=None):
    wave = AcousticWave(dictionary=deepcopy(model_dictionary))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.set_initial_velocity_model(constant=velocity)
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    return wave


def test_functional_computations_match_per_timestep_and_after_solve():
    model_dictionary = _base_functional_dictionary()

    observed_wave = _build_acoustic_wave(model_dictionary, velocity=2.5)
    observed_wave.forward_solve()
    observed_shot = np.array(observed_wave.receivers_output, copy=True)

    wave_per_timestep = _build_acoustic_wave(
        model_dictionary,
        velocity=3.0,
        real_shot_record=observed_shot,
    )
    wave_per_timestep.enable_compute_functional(
        mode=FunctionalEvaluationMode.PER_TIMESTEP
    )
    wave_per_timestep.forward_solve()

    wave_after_solve = _build_acoustic_wave(
        model_dictionary,
        velocity=3.0,
        real_shot_record=observed_shot,
    )
    wave_after_solve.enable_compute_functional(
        mode=FunctionalEvaluationMode.AFTER_SOLVE
    )
    wave_after_solve.forward_solve()

    assert wave_per_timestep.functional_value > 0.0
    assert np.allclose(
        wave_per_timestep.forward_solution_receivers,
        wave_after_solve.forward_solution_receivers,
    )
    assert np.isclose(
        wave_per_timestep.functional_value,
        wave_after_solve.functional_value,
    )
