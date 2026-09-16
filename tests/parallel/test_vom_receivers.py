from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.helpers import (
    _global_receiver_step_to_vom,
    _global_receiver_values_from_vom,
)
from spyro.utils.typing import FunctionalEvaluationMode


def _distributed_receiver_space_and_field():
    comm = fire.COMM_WORLD
    mesh = fire.UnitSquareMesh(4, 4, comm=comm)
    V = fire.FunctionSpace(mesh, "CG", 1)
    x, y = fire.SpatialCoordinate(mesh)
    field = fire.Function(V).interpolate(x + 10.0 * y)
    receiver_locations = [
        (0.1, 0.1),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.8, 0.8),
        (0.5, 0.5),
    ]
    vom = fire.VertexOnlyMesh(mesh, receiver_locations, redundant=True)
    receiver_space = fire.FunctionSpace(vom, "DG", 0)
    return receiver_space, field


def _small_vom_spatial_model():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "method": "MLT",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {
            "type": "spatial",
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
            "use_vertex_only_mesh": True,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.002,
            "dt": 0.001,
            "amplitude": 1,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def _build_small_vom_spatial_wave():
    wave = AcousticWave(_small_vom_spatial_model())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_initial_velocity_model(constant=2.0)
    return wave


@pytest.mark.parallel(2)
def test_vom_receiver_values_return_in_global_input_order():
    receiver_space, field = _distributed_receiver_space_and_field()
    local_receivers = fire.assemble(fire.interpolate(field, receiver_space))
    comm = SimpleNamespace(comm=fire.COMM_WORLD)

    receiver_values = _global_receiver_values_from_vom([local_receivers], comm)

    assert receiver_values.shape == (1, 5)
    assert np.allclose(receiver_values[0], [1.1, 8.2, 2.8, 8.8, 5.5])


@pytest.mark.parallel(2)
def test_global_receiver_step_projects_to_local_vom_space():
    receiver_space, _ = _distributed_receiver_space_and_field()
    global_record = np.array([11.0, 22.0, 33.0, 44.0, 55.0])
    comm = SimpleNamespace(comm=fire.COMM_WORLD)

    local_record = _global_receiver_step_to_vom(global_record, receiver_space)
    round_tripped = _global_receiver_values_from_vom([local_record], comm)

    assert np.allclose(round_tripped[0], global_record)


@pytest.mark.parallel(2)
def test_vom_forward_solve_with_spatial_parallelism_returns_global_record():
    wave = _build_small_vom_spatial_wave()

    wave.forward_solve()

    assert wave.forward_solution_receivers.shape == (3, 3)


@pytest.mark.parallel(2)
def test_vom_per_timestep_functional_uses_local_receiver_space():
    observed_wave = _build_small_vom_spatial_wave()
    observed_wave.forward_solve()

    wave = _build_small_vom_spatial_wave()
    wave.real_shot_record = np.array(
        observed_wave.forward_solution_receivers, copy=True
    )
    wave.enable_compute_functional(mode=FunctionalEvaluationMode.PER_TIMESTEP)

    wave.forward_solve()

    assert wave.forward_solution_receivers.shape == (3, 3)
    assert np.isclose(wave.functional_value, 0.0)
