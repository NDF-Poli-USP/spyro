from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest

from spyro.domains.space import create_function_space
from spyro.solvers.helpers import (
    _global_receiver_step_to_vom,
    _global_receiver_values_from_vom,
)


def _extruded_receiver_space_and_field():
    base = fire.UnitSquareMesh(4, 4, quadrilateral=True, comm=fire.COMM_WORLD)
    mesh = fire.ExtrudedMesh(base, 4, layer_height=0.25)
    V = create_function_space(mesh, "spectral_quadrilateral", 4, dim=3)
    x, y, z = fire.SpatialCoordinate(mesh)
    field = fire.Function(V).interpolate(fire.as_vector((
        x + 10.0 * y + 100.0 * z,
        2.0 * x + y,
        z - x,
    )))
    receiver_locations = [
        (0.1, 0.1, 0.1),
        (0.2, 0.8, 0.2),
        (0.8, 0.2, 0.4),
        (0.8, 0.8, 0.6),
        (0.5, 0.5, 0.8),
    ]
    vom = fire.VertexOnlyMesh(mesh, receiver_locations, redundant=True)
    receiver_space = create_function_space(vom, "DG0", 0, dim=3)
    return receiver_space, field, receiver_locations


@pytest.mark.parallel(4)
def test_vector_vom_values_preserve_input_order_on_extruded_mesh():
    receiver_space, field, locations = _extruded_receiver_space_and_field()
    local_receivers = fire.assemble(fire.interpolate(field, receiver_space))
    comm = SimpleNamespace(comm=fire.COMM_WORLD)

    values = _global_receiver_values_from_vom(
        [local_receivers], receiver_space, comm
    )
    expected = np.array([
        (x + 10.0 * y + 100.0 * z, 2.0 * x + y, z - x)
        for x, y, z in locations
    ])

    assert values.shape == (1, 5, 3)
    assert np.allclose(values[0], expected)


@pytest.mark.parallel(4)
def test_vector_vom_global_record_round_trip_on_extruded_mesh():
    receiver_space, _, _ = _extruded_receiver_space_and_field()
    global_record = np.arange(15, dtype=float).reshape(5, 3)
    comm = SimpleNamespace(comm=fire.COMM_WORLD)

    local_record = _global_receiver_step_to_vom(
        global_record, receiver_space
    )
    round_tripped = _global_receiver_values_from_vom(
        [local_record.dat.data_ro], receiver_space, comm
    )

    assert np.allclose(round_tripped[0], global_record)
