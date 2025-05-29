import pytest

from spyro.examples.elastic_local_abc import build_solver

# This value was obtained empirically. It is supposed for backward compatibility
expected_mechanical_energy = 0.25


def has_sufficient_memory():
    meminfo = {}
    with open('/proc/meminfo') as f:
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                meminfo[parts[0].strip()] = parts[1].strip()
    total_memory_kb = int(meminfo.get('MemTotal', '0 kB').split()[0])
    total_memory_gb = total_memory_kb / 1024 / 1024
    print(f"Total system memory {total_memory_gb}")
    return total_memory_gb > 16


@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_stacey_abc():
    wave = build_solver("Stacey", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_clayton_engquist_abc():
    wave = build_solver("CE_A1", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_central():
    wave = build_solver("Stacey", "central")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_backward_2nd():
    wave = build_solver("Stacey", "backward_2nd")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy
