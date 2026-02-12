import math
import numpy as np
from copy import deepcopy
from firedrake import *
import spyro

from tests.on_one_core.model import dictionary as model
model["acquisition"]["source_type"] = "MMS"


def run_solve(model):
    testmodel = deepcopy(model)

    Wave_obj = spyro.AcousticWaveMMS(dictionary=testmodel)
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    Wave_obj.set_initial_velocity_model(expression="1 + sin(pi*-z)*sin(pi*x)")
    Wave_obj.forward_solve()

    u_an = Wave_obj.analytical_solution(Wave_obj.current_time)
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


def test_isotropic_wave_2D():
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
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    wave.forward_solve()

    u_an = Function(wave.function_space)
    x = wave.get_spatial_coordinates()
    t = wave.current_time
    u_an.interpolate(u(x, t))

    e1 = errornorm(wave.u_n.sub(0), u_an.sub(0))
    e2 = errornorm(wave.u_n.sub(1), u_an.sub(1))

    assert math.isclose(e1, 0.0, abs_tol=1e-7)
    assert math.isclose(e2, 0.0, abs_tol=1e-7)


def test_initial_conditions_elastic():

    h = 0.25
    dt = 0.001
    # Acoustic-like solution: both components have the same scalar solution
    scalar_solution = lambda x, t: x[0] * (x[0] + 1) * x[1] * (x[1] - 1) * t
    u1 = lambda x, t: scalar_solution(x, t)
    u2 = lambda x, t: scalar_solution(x, t)
    u = lambda x, t: as_vector([u1(x, t), u2(x, t)])

    # Body forces for acoustic-like case (both components get the same source)
    scalar_source = lambda x, t: -(x[0]**2 + x[0] + x[1]**2 - x[1]) * 2 * t
    b1 = lambda x, t: scalar_source(x, t)
    b2 = lambda x, t: scalar_source(x, t)
    b = lambda x, t: as_vector([b1(x, t), b2(x, t)])

    fo = int(0.1/dt)

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
        "mesh_file": None,  # specify the mesh file
    }
    dictionary["acquisition"] = {
        "source_type": "MMS",
        "source_locations": [(-1.0, 1.0)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": [(-0.0, 0.5)],
        "body_forces": b,
    }
    dictionary["time_axis"] = {
        "initial_condition": u,
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.0,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": fo,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": False,
        "output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    dictionary["synthetic_data"] = {
        "type": "object",
        "density": 1,
        "lambda": 1,
        "mu": 0,  # Set to 0 to make elastic equivalent to acoustic
        "real_velocity_file": None,
    }
    dictionary["boundary_conditions"] = [
        ("uz", "on_boundary", 0),
        ("ux", "on_boundary", 0),
    ]

    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": h})
    # Lets build de matrix operator outside of the forward solve so we can
    # set previous timesteps for the MMS problem
    wave._initialize_model_parameters()
    wave.matrix_building()
    x = wave.get_spatial_coordinates()
    V = wave.function_space
    test_function = Function(V)

    # testing u_nm1
    test_function.interpolate(u(x, 0.0 - 2*dt))
    max_test = np.amax(test_function.dat.data[:])
    min_test = np.amin(test_function.dat.data[:])
    max_func = np.amax(wave.u_nm1.dat.data[:])
    min_func = np.amin(wave.u_nm1.dat.data[:])
    passed = math.isclose(max_test, max_func) and math.isclose(min_test, min_func)
    print(f"Test of elastic MMS u_nm1 got correct starting value: {passed}", flush=True)
    assert passed

    # testing u_n
    test_function.interpolate(u(x, 0.0 - dt))
    max_test = np.amax(test_function.dat.data[:])
    min_test = np.amin(test_function.dat.data[:])
    max_func = np.amax(wave.u_n.dat.data[:])
    min_func = np.amin(wave.u_n.dat.data[:])
    passed = math.isclose(max_test, max_func) and math.isclose(min_test, min_func)
    print(f"Test of elastic MMS u_n got correct starting value: {passed}", flush=True)
    assert passed


if __name__ == "__main__":
    test_method_triangles_lumped()
    test_method_quads_lumped()
    test_initial_conditions_elastic()
    test_isotropic_wave_2D()

    print("END")
