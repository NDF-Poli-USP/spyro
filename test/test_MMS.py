import math
from copy import deepcopy
import pytest
from firedrake import *
import spyro

from .model import model


@pytest.fixture(params=["triangle", "tetrahedral","square"])
def mesh_type(request):
    return request.param


@pytest.fixture
def mesh(mesh_type):
    if mesh_type == "triangle":
        model["opts"]["dimension"] = 2
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (0.0, 1.0), (0.0, 0.9), 256
        )
        model["acquisition"]["source_pos"] = [(-0.05, 1.5)]
    elif mesh_type == "square":
        model["opts"]['quadrature'] == 'GLL'
        model["opts"]["dimension"] = 2
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (0.0, 1.0), (0.0, 0.9), 256
        )
        model["acquisition"]["source_pos"] = [(-0.05, 1.5)]
    elif mesh_type == "tetrahedral":
        model["opts"]["dimension"] = 3
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 256
        )
        model["acquisition"]["source_pos"] = [(-0.05, 1.5, 1.5)]

    return {
        "triangle": lambda n: UnitSquareMesh(2 ** n, 2 ** n),
        "tetrahedral": lambda n: UnitCubeMesh(2 ** n, 2 ** n, 2 ** n),
        "square": lambda n: UnitSquareMesh(2 ** n, 2 ** n, quadrilateral=True),
    }[mesh_type]


@pytest.fixture(params=["CG", "KMV"])
def method_type(request):
    return request.param


@pytest.fixture
def spatial_method(method_type):
    model["opts"]["method"] = method_type
    return method_type


@pytest.fixture(params=["central", "ssprk"])
def timestep_method_type(request):
    return request.param


@pytest.fixture
def timestep_method(timestep_method_type):
    return timestep_method_type


@pytest.fixture
def interpolation_expr(mesh_type):
    return {
        "square": lambda x, y: (0.10 ** 2) * sin(pi * x) * sin(pi * y),
        "triangle": lambda x, y: (0.10 ** 2) * sin(pi * x) * sin(pi * y),
        "tetrahedral": lambda x, y, z: (0.10 ** 2) * sin(pi * x) * sin(pi * y) * sin(pi * z),
    }[mesh_type]


def run_solve(timestep_method, method, model, mesh, expr):
    testmodel = deepcopy(model)
    cell_geometry = mesh.ufl_cell()
    if method == "CG" or method == 'spectral':
        if cell_geometry == quadrilateral or cell_geometry == hexahedron:
            variant = "spectral"
            testmodel["opts"]["quadrature"] = "GLL"
        else:
            variant = "equispaced"
    elif method == "KMV":
        variant = "KMV"

    comm = spyro.utils.mpi_init(testmodel)

    element = FiniteElement(method, mesh.ufl_cell(), degree=1, variant=variant)
    V = FunctionSpace(mesh, element)

    excitation = spyro.Sources(testmodel, mesh, V, comm)

    wavelet = spyro.full_ricker_wavelet(dt=0.001, tf=1.0, freq=2.0)

    receivers = spyro.Receivers(testmodel, mesh, V, comm)

    if timestep_method == "central":
        p, _ = spyro.solvers.forward(
            testmodel, mesh, comm, Constant(1.0), excitation, wavelet, receivers
        )
    elif timestep_method == "ssprk":
        p, _ = spyro.solvers.SSPRK3(
            testmodel, mesh, comm, Constant(1.0), excitation, receivers
        )
    expr = expr(*SpatialCoordinate(mesh))
    return errornorm(interpolate(expr, V), p[-1])


def test_method(mesh, timestep_method, spatial_method, interpolation_expr):
    if mesh(3).ufl_cell() == quadrilateral and spatial_method == "KMV":
        pytest.skip("KMV isn't possible in quadrilaterals.")
    if timestep_method == "ssprk":
        pytest.skip("KMV is not yet supported in ssprk")
    error = run_solve(
        timestep_method, spatial_method, model, mesh(3), interpolation_expr
    )
    assert math.isclose(error, 0.0, abs_tol=1e-1)
