import pytest

from spyro.examples import elastic_local_abc

# This value was obtained empirically. It is supposed for backward compatibility
expected_mechanical_energy = 0.25


@pytest.fixture(autouse=True)
def isolate_output_files(tmp_path, monkeypatch):
    """Keep logger output independent of the checkout and other tests."""
    monkeypatch.setattr(elastic_local_abc, "output_dir", str(tmp_path))


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


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_stacey_abc():
    wave = elastic_local_abc.build_solver("Stacey", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_clayton_engquist_abc():
    wave = elastic_local_abc.build_solver("CE_A1", "backward")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_central():
    wave = elastic_local_abc.build_solver("Stacey", "central")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy


@pytest.mark.slow
@pytest.mark.skipif(not has_sufficient_memory(), reason="Insufficient memory")
def test_with_backward_2nd():
    wave = elastic_local_abc.build_solver("Stacey", "backward_2nd")
    wave.forward_solve()
    last_mechanical_energy = wave.field_logger.get("mechanical_energy")
    assert last_mechanical_energy < expected_mechanical_energy
