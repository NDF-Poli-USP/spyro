import math
from copy import deepcopy

import pytest
from firedrake import *

import spyro

from .model import model


@pytest.fixture(params=["square", "cube"])
def mesh_type(request):
    return request.param


@pytest.fixture
def mesh(mesh_type):
    if mesh_type == "square":
        model["opts"]["element"] = "tria"
        model["opts"]["dimension"] = 2
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (0.0, 1.0), (0.0, 0.9), 256
        )
        model["acquisition"]["source_pos"] = [(-0.05, 1.5)]
    elif mesh_type == "cube":
        model["opts"]["element"] = "tetra"
        model["opts"]["dimension"] = 3
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 256
        )
        model["acquisition"]["source_pos"] = [(-0.05, 1.5, 1.5)]
    return {
        "square": lambda n: UnitSquareMesh(2 ** n, 2 ** n),
        "cube": lambda n: UnitCubeMesh(2 ** n, 2 ** n, 2 ** n),
    }[mesh_type]


@pytest.fixture(params=["CG", "KMV"])
def method_type(request):
    return request.param


@pytest.fixture
def spatial_method(method_type):
    model["opts"]["method"] = method_type
    return method_type


@pytest.fixture(params=["leapfrog", "ssprk"])
def timestep_method_type(request):
    return request.param


@pytest.fixture
def timestep_method(timestep_method_type):
    return timestep_method_type


@pytest.fixture
def interpolation_expr(mesh_type):
    return {
        "square": lambda x, y: (0.10 ** 2) * sin(pi * x) * sin(pi * y),
        "cube": lambda x, y, z: (0.10 ** 2) * sin(pi * x) * sin(pi * y) * sin(pi * z),
    }[mesh_type]


def run_solve(timestep_method, method, model, mesh, expr):
    testmodel = deepcopy(model)
    if method == "KMV":
        variant = "KMV"
        testmodel["opts"]["quadrature"] = "KMV"
    else:
        variant = "equispaced"

    comm = spyro.utils.mpi_init(testmodel)

    element = FiniteElement(method, mesh.ufl_cell(), degree=1, variant=variant)
    V = FunctionSpace(mesh, element)

    excitation = spyro.Sources(testmodel, mesh, V, comm).create()
    receivers = spyro.Receivers(testmodel, mesh, V, comm).create()

    if timestep_method == "leapfrog":
        p, _ = spyro.solvers.Leapfrog(
            testmodel, mesh, comm, Constant(1.0), excitation, receivers
        )
    elif timestep_method == "ssprk":
        p, _ = spyro.solvers.SSPRK3(
            testmodel, mesh, comm, Constant(1.0), excitation, receivers
        )
    expr = expr(*SpatialCoordinate(mesh))
    return errornorm(interpolate(expr, V), p[-1])


def test_method(mesh, timestep_method, spatial_method, interpolation_expr):
    if spatial_method == "KMV" and timestep_method == "ssprk":
        pytest.skip("KMV is not yet supported in ssprk")
    error = run_solve(
        timestep_method, spatial_method, model, mesh(3), interpolation_expr
    )
    assert math.isclose(error, 0.0, abs_tol=1e-1)
