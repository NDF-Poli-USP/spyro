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

    u_an = Wave_obj.analytical_solution(Wave_obj.current_time)
    u_num = Wave_obj.u_n

    return errornorm(u_num, u_an)


def test_method(mesh_type, method_type):
    error = run_solve(model)

    assert math.isclose(error, 0.0, abs_tol=1e-7)

def test_isotropic_wave_2D(mesh_type, method_type):
    u1 = lambda x, t: (x[0]**2 + x[0])*(x[1]**2 - x[1])*t
    u2 = lambda x, t: (2*x[0]**2 + 2*x[0])*(-x[1]**2 + x[1])*t
    u = lambda x, t: as_vector([u1(x, t), u2(x, t)])

    b1 = lambda x, t: -(2*x[0]**2 + 6*x[1]**2 - 16*x[0]*x[1] + 10*x[0] - 14*x[1] + 4)*t
    b2 = lambda x, t: -(-12*x[0]**2 - 4*x[1]**2 + 8*x[0]*x[1] - 16*x[0] + 8*x[1] - 2)*t
    b = lambda x, t: as_vector([b1(x, t), b2(x, t)])
    
    dt = 1e-3
    fo = int(0.1/dt)

    d = deepcopy(model)
    d["acquisition"]["body_forces"] = b
    d["time_axis"]["initial_condition"] = u
    d["synthetic_data"] = {
        "type": "object",
        "density": 1,
        "lambda": 1,
        "mu": 1,
        "real_velocity_file": None,
    }
    d["time_axis"]["dt"] = dt
    d["time_axis"]["output_frequency"] = fo
    d["boundary_conditions"] = [
        ("uz", "on_boundary", 0),
        ("ux", "on_boundary", 0),
    ]

    wave = spyro.IsotropicWave(d)
    wave.set_mesh(mesh_parameters={"dx": 0.02})
    wave.forward_solve()

    u_an = Function(wave.function_space)
    x = wave.get_spatial_coordinates()
    t = wave.current_time
    u_an.interpolate(u(x, t))

    e1 = errornorm(wave.u_n.sub(0), u_an.sub(0))
    e2 = errornorm(wave.u_n.sub(1), u_an.sub(1))

    assert math.isclose(e1, 0.0, abs_tol=1e-7)
    assert math.isclose(e2, 0.0, abs_tol=1e-7)