import pytest

from spyro.examples.elastic_local_abc import build_solver
from spyro.mpi.spyro_mpi import SpyroEnsemble

# This value was obtained empirically. It is supposed for backward compatibility
expected_mechanical_energy = 0.25


def has_sufficient_memory(threshold_gb=16):
    """Return True if available RAM plus free SWAP exceeds threshold_gb."""
    meminfo = {}
    with open('/proc/meminfo') as f:
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                meminfo[parts[0].strip()] = parts[1].strip()
    total_memory_gb = int(meminfo.get('MemTotal', '0 kB').split()[0]) / 1024**2
    total_mswap_gb = int(meminfo.get('SwapTotal', '0 kB').split()[0]) / 1024**2
    SpyroEnsemble.print(f"Total system memory: {total_memory_gb + total_mswap_gb: .2f} GB")
    return total_memory_gb > threshold_gb


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_stacey_abc():
    wave = build_solver("Stacey", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_clayton_engquist_abc():
    wave = build_solver("CE_A1", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_central():
    wave = build_solver("Stacey", "central")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_backward_2nd():
    wave = build_solver("Stacey", "backward_2nd")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy
