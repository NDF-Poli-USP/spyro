"""Serial unit tests for the MPI helpers in `spyro.mpi.spyro_mpi`.

The tests cover two parts of the module:

- The parallelism configurations (`ParallelismConfig`, `CustomParallelism`,
  `AutomaticParallelism` and `SpatialParallelism`): which shots each
  propagation runs and how many cores each ensemble member gets.
- The `SpyroEnsemble` helpers that behave differently on each rank: printing,
  reductions, VTK output, and the decorators that run a function on one rank
  and broadcast its result.

The tests run on a single core. `COMM_WORLD` and the Firedrake ensemble are
replaced by mocks, so each rank-dependent branch can be checked by pretending
to be a given rank, without launching MPI. The parallel behaviour itself is
tested under `tests/parallel`.
"""
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
    """Restore the ensemble state of `SpyroEnsemble` after each test.

    `SpyroEnsemble.ensemble` and `SpyroEnsemble.config` are class attributes,
    so they are shared by every test in the pytest process. Several tests in
    this module set them directly to mocks to pretend to be a given rank.
    Without this fixture, a test in another module could later find a
    `MagicMock` ensemble and take the wrong branch, for example printing or
    writing files only "on rank 0" of a fake communicator. The fixture is
    `autouse`, so it wraps every test in this module.

    Yields
    ------
    None
        Control to the test. Afterwards, the values saved before the test
        are put back.
    """
    ensemble, config = SpyroEnsemble.ensemble, SpyroEnsemble.config
    yield
    SpyroEnsemble.ensemble, SpyroEnsemble.config = ensemble, config


def test_parallelism_config_shot_ids():
    """By default, each propagation runs a single shot, in order."""
    config = ParallelismConfig(number_of_sources=3)

    assert config.shot_ids_per_propagation == [
        [0],
        [1],
        [2],
    ]


def test_custom_parallelism_shot_ids():
    """Custom parallelism uses the shot groups given by the user as they are."""
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
    """Automatic parallelism splits the cores evenly between the shots.

    With 8 cores and 2 shots, each ensemble member gets 4 cores.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` with an 8-core communicator.
    """
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 8})(),
    )

    config = AutomaticParallelism(number_of_sources=2)

    assert config.get_number_of_cores() == 4


def test_automatic_parallelism_requires_even_division(monkeypatch):
    """Automatic parallelism raises if the cores do not split evenly.

    8 cores cannot be shared evenly by 3 shots, so `ValueError` is raised.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` with an 8-core communicator.
    """
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 8})(),
    )

    config = AutomaticParallelism(number_of_sources=3)

    with pytest.raises(ValueError):
        config.get_number_of_cores()


def test_spatial_parallelism_uses_all_cores(monkeypatch):
    """Spatial parallelism gives every core to a single ensemble member.

    The shots run one after another, each over all 16 cores.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` with a 16-core communicator.
    """
    monkeypatch.setattr(
        "spyro.mpi.spyro_mpi.COMM_WORLD",
        type("MockComm", (), {"size": 16})(),
    )

    config = SpatialParallelism(number_of_sources=4)

    assert config.get_number_of_cores() == 16


def test_custom_parallelism_cores(monkeypatch):
    """Custom parallelism splits the cores evenly between the shot groups.

    With 12 cores and 4 shot groups, each ensemble member gets 3 cores.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` with a 12-core communicator.
    """
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
    """Before the ensemble is initialized, `SpyroEnsemble.print` always prints.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Fixture that captures standard output.
    """
    SpyroEnsemble.ensemble = None

    SpyroEnsemble.print("Hello")

    captured = capsys.readouterr()

    assert captured.out == "Hello\n"


def test_print_when_ensemble_is_not_none_and_rank_is_zero(capsys):
    """With an ensemble, global rank 0 prints the message.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Fixture that captures standard output.
    """
    SpyroEnsemble.ensemble = MagicMock()

    global_comm = MagicMock()
    global_comm.rank = 0
    SpyroEnsemble.ensemble.global_comm = global_comm

    SpyroEnsemble.print("test")

    assert capsys.readouterr().out == "test\n"


def test_print_when_ensemble_is_not_none_and_rank_is_one(capsys):
    """With an ensemble, ranks other than global rank 0 stay silent.

    This keeps the output from being repeated once per rank.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Fixture that captures standard output.
    """
    SpyroEnsemble.ensemble = MagicMock()

    global_comm = MagicMock()
    global_comm.rank = 1
    SpyroEnsemble.ensemble.global_comm = global_comm

    SpyroEnsemble.print("test")

    assert capsys.readouterr().out == ""


def test_communicate_when_size_is_one():
    """With one spatial rank, `communicate` returns its input unchanged.

    There is nothing to reduce, so no MPI call is needed.
    """
    SpyroEnsemble.ensemble = MagicMock()

    comm = MagicMock()
    comm.size = 1
    SpyroEnsemble.ensemble.comm = comm

    reduced = SpyroEnsemble.communicate([1])

    assert reduced == [1]


def test_communicate_when_size_is_ten():
    """With several spatial ranks, `communicate` returns the reduced values.

    The spatial communicator is a mock with 10 ranks, so the reduction
    branch runs. The values are the same on every rank here, so the
    reduction must give them back unchanged.
    """
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
    """Install a mocked ensemble on `SpyroEnsemble`.

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
    """The first ensemble member writes the VTK file on its spatial comm.

    The keyword arguments reach `VTKFile.write` unchanged.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `VTKFile` with a mock.
    """
    ensemble = _mock_ensemble_with_rank(0)
    vtk_file = MagicMock()
    monkeypatch.setattr("spyro.mpi.spyro_mpi.VTKFile", vtk_file)
    field = object()

    SpyroEnsemble.write_vtk("out.pvd", field, name="velocity")

    vtk_file.assert_called_once_with("out.pvd", comm=ensemble.comm)
    vtk_file.return_value.write.assert_called_once_with(field, name="velocity")


def test_write_vtk_skipped_on_other_ensemble_members(monkeypatch):
    """Ensemble members other than the first do not write the VTK file.

    Every member holds the same fields, so writing them once is enough,
    and it avoids members racing on the same files.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `VTKFile` with a mock.
    """
    _mock_ensemble_with_rank(1)
    vtk_file = MagicMock()
    monkeypatch.setattr("spyro.mpi.spyro_mpi.VTKFile", vtk_file)

    SpyroEnsemble.write_vtk("out.pvd", object())

    vtk_file.assert_not_called()


class BroadcastComm:
    """Stand-in for `COMM_WORLD` that records broadcasts.

    Attributes
    ----------
    rank : int
        Rank of this process.
    broadcasts : list
        Values passed to `bcast`.
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
            `value` on the root, `root_value` elsewhere.
        """
        self.broadcasts.append(value)
        return value if self.rank == root else self.root_value


def test_run_in_one_core_and_broadcast_without_ensemble_runs_on_rank_0(
    monkeypatch,
):
    """Before the ensemble exists, rank 0 runs the function, with kwargs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` and the ensemble.
    """
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
    """Before the ensemble exists, other ranks receive rank 0's result.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `COMM_WORLD` and the ensemble.
    """
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
    """Keyword arguments reach the function as keywords.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the ensemble with a mock.
    """
    ensemble = MagicMock()
    ensemble.global_comm.rank = 0
    monkeypatch.setattr(SpyroEnsemble, "ensemble", ensemble)

    @SpyroEnsemble.run_in_one_core
    def subtract(a, b=0):
        return a - b

    assert subtract(5, b=2) == 3
