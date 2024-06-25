import math
from copy import deepcopy
import pytest
from firedrake import *
import spyro

from .model import dictionary as model

model["acquisition"]["source_type"] = "MMS"


@pytest.fixture(params=["triangle", "square"])
def mesh_type(request):
    if mesh_type == "triangle":
        model["cell_type"] = "triangles"
    elif mesh_type == "square":
        model["cell_type"] = "quadrilaterals"
    return request.param


@pytest.fixture(params=["lumped", "equispaced"])
def method_type(request):
    if method_type == "lumped":
        model["variant"] = "lumped"
    elif method_type == "equispaced":
        model["variant"] = "equispaced"
    return request.param


def run_solve(model):
    testmodel = deepcopy(model)

    Wave_obj = spyro.AcousticWaveMMS(dictionary=testmodel)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.02})
    Wave_obj.set_initial_velocity_model(expression="1 + sin(pi*-z)*sin(pi*x)")
    Wave_obj.forward_solve()

    u_an = Wave_obj.analytical
    u_num = Wave_obj.u_n

    return errornorm(u_num, u_an)


def test_method(mesh_type, method_type):
    error = run_solve(model)

    assert math.isclose(error, 0.0, abs_tol=1e-7)
