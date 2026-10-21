from unittest.mock import MagicMock

import pytest

from spyro.mpi.spyro_mpi import (
    ParallelismConfig,
    AutomaticParallelism,
    SpatialParallelism,
    CustomParallelism,
)
from spyro.mpi.spyro_mpi import SpyroEnsemble


@pytest.fixture(autouse=True)
def restore_spyro_ensemble():
    """Restore the class-level ensemble state that tests here replace.

    Yields
    ------
    None
        Control to the test, then the original state is put back so mocks
        do not leak into tests in other modules.
    """
    ensemble, config = SpyroEnsemble.ensemble, SpyroEnsemble.config
    yield
    SpyroEnsemble.ensemble, SpyroEnsemble.config = ensemble, config


def test_parallelism_config_shot_ids():
    config = ParallelismConfig(number_of_sources=3)

    assert config.shot_ids_per_propagation == [
        [0],
        [1],
        [2],
    ]


def test_custom_parallelism_shot_ids():
    config = CustomParallelism(
        number_of_sources=4,
        custom_shot_ids=[[0, 1], [2], [3]],
    )

    assert config.shot_ids_per_propagation == [
        [0, 1],
        [2],
        [3],
    ]


def test_automatic_parallelism_cores(monkeypatch):
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 8})(),
    )

    config = AutomaticParallelism(number_of_sources=2)

    assert config.get_number_of_cores() == 4


def test_automatic_parallelism_requires_even_division(monkeypatch):
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 8})(),
    )

    config = AutomaticParallelism(number_of_sources=3)

    with pytest.raises(ValueError):
        config.get_number_of_cores()


def test_spatial_parallelism_uses_all_cores(monkeypatch):
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 16})(),
    )

    config = SpatialParallelism(number_of_sources=4)

    assert config.get_number_of_cores() == 16


def test_custom_parallelism_cores(monkeypatch):
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 12})(),
    )

    config = CustomParallelism(
        number_of_sources=6,
        custom_shot_ids=[[0, 1], [2], [3, 4], [5]],
    )

    assert config.get_number_of_cores() == 3


def test_print_when_ensemble_is_none(capsys):
    SpyroEnsemble.ensemble = None

    SpyroEnsemble.print("Hello")

    captured = capsys.readouterr()

    assert captured.out == "Hello\n"


def test_print_when_ensemble_is_not_none_and_rank_is_zero(capsys):
    SpyroEnsemble.ensemble = MagicMock()

    global_comm = MagicMock()
    global_comm.rank = 0
    SpyroEnsemble.ensemble.global_comm = global_comm

    SpyroEnsemble.print("test")

    assert capsys.readouterr().out == "test\n"


def test_print_when_ensemble_is_not_none_and_rank_is_one(capsys):
    SpyroEnsemble.ensemble = MagicMock()

    global_comm = MagicMock()
    global_comm.rank = 1
    SpyroEnsemble.ensemble.global_comm = global_comm

    SpyroEnsemble.print("test")

    assert capsys.readouterr().out == ""


def test_communicate_when_size_is_one():
    SpyroEnsemble.ensemble = MagicMock()

    comm = MagicMock()
    comm.size = 1
    SpyroEnsemble.ensemble.comm = comm

    reduced = SpyroEnsemble.communicate([1])

    assert reduced == [1]


def test_communicate_when_size_is_ten():
    SpyroEnsemble.ensemble = MagicMock()

    comm = MagicMock()
    comm.size = 10

    global_comm = MagicMock()
    global_comm.rank = 1

    def fake_allreduce(s, r, op):
        r[:] = s

    comm.AllReduce.side_effect = fake_allreduce
    SpyroEnsemble.ensemble.global_comm = global_comm
    SpyroEnsemble.ensemble.comm = comm

    reduced = SpyroEnsemble.communicate([1, 4])

    assert reduced == [1, 4]


def _mock_ensemble_with_rank(ensemble_rank: int) -> MagicMock:
    """Install a mocked ensemble on ``SpyroEnsemble``.

    Parameters
    ----------
    ensemble_rank : int
        Rank of this process in the ensemble communicator.

    Returns
    -------
    MagicMock
        The mocked ensemble.
    """
    SpyroEnsemble.ensemble = MagicMock()
    SpyroEnsemble.ensemble.ensemble_comm.rank = ensemble_rank
    return SpyroEnsemble.ensemble


def test_write_vtk_on_first_ensemble_member(monkeypatch):
    ensemble = _mock_ensemble_with_rank(0)
    vtk_file = MagicMock()
    monkeypatch.setattr("spyro.mpi.spyro_mpi.VTKFile", vtk_file)
    field = object()

    SpyroEnsemble.write_vtk("out.pvd", field, name="velocity")

    vtk_file.assert_called_once_with("out.pvd", comm=ensemble.comm)
    vtk_file.return_value.write.assert_called_once_with(field, name="velocity")


def test_write_vtk_skipped_on_other_ensemble_members(monkeypatch):
    _mock_ensemble_with_rank(1)
    vtk_file = MagicMock()
    monkeypatch.setattr("spyro.mpi.spyro_mpi.VTKFile", vtk_file)

    SpyroEnsemble.write_vtk("out.pvd", object())

    vtk_file.assert_not_called()


class BroadcastComm:
    """Stand-in for ``COMM_WORLD`` that records broadcasts.

    Attributes
    ----------
    rank : int
        Rank of this process.
    broadcasts : list
        Values passed to :meth:`bcast`.
    root_value : object
        Value received by ranks other than the root.
    """

    def __init__(self, rank: int, root_value: object = None) -> None:
        """Set the rank and the value the root broadcasts.

        Parameters
        ----------
        rank : int
            Rank of this process.
        root_value : object, optional
            Value received by ranks other than the root. Default is None.
        """
        self.rank = rank
        self.broadcasts = []
        self.root_value = root_value

    def bcast(self, value: object, root: int = 0) -> object:
        """Return the root's value, as an MPI broadcast does.

        Parameters
        ----------
        value : object
            Value sent by this rank.
        root : int, optional
            Broadcasting rank. Default is 0.

        Returns
        -------
        object
            ``value`` on the root, ``root_value`` elsewhere.
        """
        self.broadcasts.append(value)
        return value if self.rank == root else self.root_value


def test_run_in_one_core_and_broadcast_without_ensemble_runs_on_rank_0(
    monkeypatch,
):
    """Before the ensemble exists, rank 0 runs the function, with kwargs."""
    comm = BroadcastComm(rank=0)
    monkeypatch.setattr("spyro.mpi.spyro_mpi.COMM_WORLD", comm)
    monkeypatch.setattr(SpyroEnsemble, "ensemble", None)
    calls = []

    @SpyroEnsemble.run_in_one_core_and_broadcast
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, b=2) == 3
    assert calls == [(1, 2)]
    assert comm.broadcasts == [3]


def test_run_in_one_core_and_broadcast_without_ensemble_skips_other_ranks(
    monkeypatch,
):
    """Before the ensemble exists, other ranks receive rank 0's result."""
    comm = BroadcastComm(rank=1, root_value=3)
    monkeypatch.setattr("spyro.mpi.spyro_mpi.COMM_WORLD", comm)
    monkeypatch.setattr(SpyroEnsemble, "ensemble", None)
    calls = []

    @SpyroEnsemble.run_in_one_core_and_broadcast
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, b=2) == 3
    assert calls == []
    assert comm.broadcasts == [None]


def test_run_in_one_core_passes_keyword_arguments(monkeypatch):
    """Keyword arguments reach the function as keywords."""
    ensemble = MagicMock()
    ensemble.global_comm.rank = 0
    monkeypatch.setattr(SpyroEnsemble, "ensemble", ensemble)

    @SpyroEnsemble.run_in_one_core
    def subtract(a, b=0):
        return a - b

    assert subtract(5, b=2) == 3
