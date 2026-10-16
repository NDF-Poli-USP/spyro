"""Vertex-only-mesh receivers under spatial parallelism (issue #315).

Run with ``mpiexec -n 2 pytest tests/parallel/test_vom_receivers.py``: the
single shot is decomposed over both ranks, so each rank owns only the
receivers inside its mesh partition.
"""
import numpy as np
import pytest

import spyro
from spyro.utils.typing import FunctionalEvaluationMode

# Scattered on purpose: for a transect the vertex-only-mesh order matches
# the input order, which would hide an ordering bug.
receiver_locations = [
    (-0.2, 0.8),
    (-0.5, 0.2),
    (-0.3, 0.6),
    (-0.6, 0.9),
    (-0.15, 0.3),
    (-0.45, 0.5),
]

dictionary = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "method": "MLT",
        "degree": 2,
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
        "source_locations": [(-0.3, 0.5)],
        "frequency": 8.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": receiver_locations,
    },
    "time_axis": {
        "initial_time": 0.0,
        "final_time": 0.3,
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


def _wave(use_vertex_only_mesh, real_shot_record=None):
    dictionary["acquisition"]["use_vertex_only_mesh"] = use_vertex_only_mesh
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
        wave.enable_compute_functional(
            mode=FunctionalEvaluationMode.PER_TIMESTEP
        )
    return wave


@pytest.mark.parallel(2)
def test_vom_receivers_spatial_parallel():
    # The Dirac-delta projector is the reference: input order, independent
    # of the mesh partition.
    dirac = _wave(use_vertex_only_mesh=False)
    dirac.forward_solve()
    expected = np.asarray(dirac.forward_solution_receivers)
    assert np.abs(expected).max() > 1e-3, "wave did not reach the receivers"

    vom = _wave(use_vertex_only_mesh=True)
    vom.forward_solve()
    record = np.asarray(vom.forward_solution_receivers)
    assert record.shape == expected.shape
    assert np.allclose(record, expected, atol=1e-10 * np.abs(expected).max())

    # Per-timestep misfit (automated-adjoint path): the observed record
    # must be restricted to the receivers each rank owns.
    same = _wave(use_vertex_only_mesh=True, real_shot_record=expected)
    same.forward_solve()
    assert same.functional_value == pytest.approx(0.0, abs=1e-20)

    permuted = _wave(
        use_vertex_only_mesh=True, real_shot_record=expected[:, ::-1].copy()
    )
    permuted.forward_solve()
    assert permuted.functional_value > 1e-8
