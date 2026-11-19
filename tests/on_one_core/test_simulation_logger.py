import firedrake as fire
import importlib
import numpy as np
import os
import pytest
import warnings
import uuid

from spyro.io.dictionaryio import Read_outputs
from spyro.io.simulation_logger import SimulationLogger
from spyro.mpi.spyro_mpi import AutomaticParallelism, SpyroEnsemble
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave


@pytest.fixture
def init_mpi():
    SpyroEnsemble.initialize(AutomaticParallelism(number_of_sources=1))


@pytest.fixture
def logger():
    mesh = fire.UnitIntervalMesh(2)
    V = fire.FunctionSpace(mesh, "DG", 0)
    u = fire.Function(V)

    rnd_str = str(uuid.uuid4())[:4]
    a_str = "asn" + rnd_str
    b_str = "bsn" + rnd_str
    c_str = "csn" + rnd_str
    d = {a_str + "_output": True, b_str + "_output": True, c_str + "_output": False}
    SpyroEnsemble.initialize(AutomaticParallelism(number_of_sources=1))
    logger = SimulationLogger(d)
    logger.rnd_str = rnd_str
    logger.add_field(a_str, "1st", lambda: u)
    logger.add_field(b_str, "2nd", lambda: u)
    logger.add_field(c_str, "3rd", lambda: u)
    return logger


def test_writing(logger):
    logger.start_logging(0)
    logger.log(0)
    rnd_str = logger.rnd_str

    assert os.path.isfile("asn" + rnd_str + "sn0.pvd")
    assert os.path.isfile("bsn" + rnd_str + "sn0.pvd")
    assert not os.path.isfile("csn" + rnd_str + "sn0.pvd")


def test_warning(logger):
    logger.start_logging(0)
    with pytest.warns(UserWarning):
        logger.start_logging(1)


def test_no_warning(logger):
    logger.start_logging(0)
    logger.stop_logging()
    # Assert that no warnings are emitted
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logger.start_logging(1)


def _energy_logger(
    visualization: dict | Read_outputs, values: list[float]
) -> SimulationLogger:
    """Build a logger whose `acoustic_energy` functional returns `values[0]`.

    Parameters
    ----------
    visualization : dict or Read_outputs
        Visualization and logging configuration given to the logger.
    values : list of float
        One-element list read by the functional callback, so that a test can change
        the sampled value between logging steps.

    Returns
    -------
    SimulationLogger
        Logger with the `acoustic_energy` functional registered.
    """
    logger = SimulationLogger(visualization)
    logger.add_functional("acoustic_energy", lambda: values[0])
    return logger


def test_list_source_ids_in_field_filename(init_mpi, tmp_path):
    # The propagator passes the source ids as a list, not as an int
    u = fire.Function(fire.FunctionSpace(fire.UnitIntervalMesh(2), "DG", 0))
    logger = SimulationLogger(
        {"forward_output": True, "forward_output_filename": f"{tmp_path}/fwd.pvd"}
    )
    logger.add_field("forward", "u", lambda: u)
    logger.start_logging([0])
    logger.log(0.0)
    logger.stop_logging()

    assert os.path.isfile(tmp_path / "fwdsn[0].pvd")


def test_dot_in_field_directory(init_mpi, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    u = fire.Function(fire.FunctionSpace(fire.UnitIntervalMesh(2), "DG", 0))
    logger = SimulationLogger(
        {"forward_output": True, "forward_output_filename": "./out/fwd.pvd"}
    )
    logger.add_field("forward", "u", lambda: u)
    logger.start_logging([0])
    logger.log(0.0)
    logger.stop_logging()

    assert os.path.isfile(tmp_path / "out" / "fwdsn[0].pvd")


def test_default_functional_filename(init_mpi, tmp_path, monkeypatch):
    # `Read_outputs` stores `None` when no energy filename is given
    monkeypatch.chdir(tmp_path)
    logger = _energy_logger(Read_outputs(acoustic_energy=True), [1.0])
    logger.start_logging([0])
    logger.log(0.0)
    logger.stop_logging()

    assert np.load(tmp_path / "acoustic_energysn[0].npy").tolist() == [1.0]


def test_functional_parent_directory_is_created(init_mpi, tmp_path):
    filename = f"{tmp_path}/new/dir/energy.npy"
    logger = _energy_logger(
        Read_outputs(acoustic_energy=True, acoustic_energy_filename=filename), [1.0]
    )
    logger.start_logging([0])
    logger.log(0.0)
    logger.stop_logging()

    assert os.path.isfile(tmp_path / "new" / "dir" / "energysn[0].npy")


def test_serial_shots_keep_their_own_functional_files(init_mpi, tmp_path):
    values = [0.0]
    logger = _energy_logger(
        Read_outputs(
            acoustic_energy=True,
            acoustic_energy_filename=f"{tmp_path}/energy.npy",
            time_filename=f"{tmp_path}/time.npy",
        ),
        values,
    )
    for shot in (0, 1):
        logger.start_logging([shot])
        for t in (0.0, 0.5):
            values[0] = 10.0 * shot + t
            logger.log(t)
        logger.stop_logging()
        assert logger.get("acoustic_energy") == 10.0 * shot + 0.5

    assert np.load(tmp_path / "energysn[0].npy").tolist() == [0.0, 0.5]
    assert np.load(tmp_path / "energysn[1].npy").tolist() == [10.0, 10.5]
    assert np.load(tmp_path / "time.npy").tolist() == [0.0, 0.5]


def test_disabled_functional_is_not_saved(init_mpi, tmp_path):
    logger = _energy_logger(
        Read_outputs(acoustic_energy_filename=f"{tmp_path}/energy.npy"), [1.0]
    )
    logger.start_logging([0])
    logger.log(0.0)
    logger.stop_logging()

    assert list(tmp_path.iterdir()) == []
    with pytest.raises(KeyError):
        logger.get("acoustic_energy")


def test_deprecated_module_alias():
    with pytest.warns(DeprecationWarning, match="simulation_logger"):
        module = importlib.import_module("spyro.io.field_logger")
        importlib.reload(module)

    assert module.FieldLogger is SimulationLogger


def test_p_and_s_wave_output(tmp_path):
    dictionary = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [[-0.5, 0.5]],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([0, 1]),
            "receiver_locations": [],
        },
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": 1.5,
            "s_wave_velocity": 1.0,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.01,
            "dt": 0.001,
            "output_frequency": 5,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "p_wave_output": True,
            "p_wave_output_filename": f"{tmp_path}/p.pvd",
            "s_wave_output": True,
            "s_wave_output_filename": f"{tmp_path}/s.pvd",
        },
    }
    wave = IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.forward_solve()

    assert os.path.isfile(tmp_path / "psn[0].pvd")
    assert os.path.isfile(tmp_path / "ssn[0].pvd")
    with pytest.warns(DeprecationWarning, match="simulation_logger"):
        assert wave.field_logger is wave.simulation_logger
