import pytest
import firedrake as fire

from spyro.domains.space import create_function_space


def test_create_function_space_dg0_on_vertex_only_mesh():
    mesh = fire.UnitSquareMesh(1, 1)
    vom = fire.VertexOnlyMesh(mesh, [[0.25, 0.25], [0.75, 0.25]])

    V = create_function_space(vom, "DG0", 0)
    W = create_function_space(vom, "DG0", 0, dim=2)

    assert V.value_size == 1
    assert V.ufl_element().degree() == 0
    assert W.value_size == 2
    assert W.ufl_element().degree() == 0


def test_create_function_space_dg0_rejects_nonzero_degree():
    mesh = fire.UnitSquareMesh(1, 1)

    with pytest.raises(ValueError, match="DG0 requires degree 0"):
        create_function_space(mesh, "DG0", 1)


def test_create_function_space_from_existing_element():
    mesh = fire.UnitSquareMesh(1, 1)
    V = create_function_space(mesh, "CG", 1)

    scalar_space = create_function_space(mesh, V.ufl_element(), None)
    vector_space = create_function_space(mesh, V.ufl_element(), None, dim=None)
    tensor_space = create_function_space(
        mesh, V.ufl_element(), None, shape=(2, 2))

    assert scalar_space.value_size == 1
    assert vector_space.value_size == 2
    assert tensor_space.value_size == 4
