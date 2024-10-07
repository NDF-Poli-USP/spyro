import firedrake as fire
import os
import pytest
import warnings

from mpi4py import MPI
from spyro.io.field_logger import FieldLogger

comm = fire.Ensemble(MPI.COMM_WORLD, 1)

@pytest.fixture
def logger():
    mesh = fire.UnitIntervalMesh(2)
    V = fire.FunctionSpace(mesh, "DG", 0)
    u = fire.Function(V)

    d = {
        "a_output": True,
        "b_output": True,
        "c_output": False
    }
    logger = FieldLogger(comm, d)
    logger.add_field("a", "1st", lambda: u)
    logger.add_field("b", "2nd", lambda: u)
    logger.add_field("c", "3rd", lambda: u)
    return logger

def test_writing(logger):
    logger.start_logging(0)
    logger.log(0)

    assert os.path.isfile("asn0.pvd")
    assert os.path.isfile("bsn0.pvd")
    assert not os.path.isfile("csn0.pvd")

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