import math
from copy import deepcopy
import pytest
from firedrake import *
import spyro

from model import dictionary as model
model["acquisition"]["source_type"] = "MMS"


def run_solve(model):
    testmodel = deepcopy(model)

    Wave_obj = spyro.AcousticWaveMMS(dictionary=testmodel)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.02})
    Wave_obj.set_initial_velocity_model(expression="1 + sin(pi*-z)*sin(pi*x)")
    Wave_obj.forward_solve()

    u_an = Wave_obj.analytical
    u_num = Wave_obj.u_n

    return errornorm(u_num, u_an)


def run_method(mesh_type, method_type):
    model["options"]["cell_type"] = mesh_type
    model["options"]["variant"] = method_type
    print(f"For {mesh_type} and {method_type}")
    error = run_solve(model)
    test = math.isclose(error, 0.0, abs_tol=1e-7)
    print(f"Error is {error}")
    print(f"Test: {test}")

    assert test


def test_method_triangles_lumped():
    run_method("triangles", "lumped")


def test_method_quads_lumped():
    run_method("quadrilaterals", "lumped")


if __name__ == "__main__":
    test_method_triangles_lumped()
    test_method_quads_lumped()

    print("END")
