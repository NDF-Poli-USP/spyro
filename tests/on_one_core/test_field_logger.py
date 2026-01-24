import firedrake as fire
import os
import pytest
import warnings
import uuid

from mpi4py import MPI
from spyro.io.field_logger import FieldLogger

comm = fire.Ensemble(MPI.COMM_WORLD, 1)


@pytest.fixture
def logger():
    mesh = fire.UnitIntervalMesh(2)
    V = fire.FunctionSpace(mesh, "DG", 0)
    u = fire.Function(V)

    rnd_str = str(uuid.uuid4())[:4]
    a_str = "asn" + rnd_str
    b_str = "bsn" + rnd_str
    c_str = "csn" + rnd_str
    d = {
        a_str + "_output": True,
        b_str + "_output": True,
        c_str + "_output": False
    }
    logger = FieldLogger(comm, d)
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
