import pytest

from spyro.examples.elastic_local_abc import build_solver

# This value was obtained empirically. It is supposed for backward compatibility
expected_mechanical_energy = 0.25


def test_stacey_abc():
    wave = build_solver("Stacey", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


def test_clayton_engquist_abc():
    wave = build_solver("CE_A1", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


def test_with_central():
    wave = build_solver("Stacey", "central")
    with pytest.raises(AssertionError) as e:
        wave.forward_solve()


def test_with_backward_2nd():
    wave = build_solver("Stacey", "backward_2nd")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy