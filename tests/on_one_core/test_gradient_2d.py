from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest
import spyro
from pyadjoint import Tape, continue_annotation, set_working_tape


STEPS = (1e-3, 1e-4, 1e-5)


def set_dictionary(automatic_adjoint):
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
        "automatic_adjoint": automatic_adjoint,
    }
    dictionary["parallelism"] = {"type": "automatic"}
    dictionary["mesh"] = {
        "Lz": 3.0,
        "Lx": 3.0,
        "Ly": 0.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.1, 1.5)],
        "frequency": 5.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": spyro.create_transect(
            (-1.8, 1.2), (-1.8, 1.8), 10
        ),
        "use_vertex_only_mesh": True,
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": 1.0,
        "dt": 0.0005,
        "amplitude": 1,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    }
    dictionary["visualization"] = {
        "forward_output": False,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": "results/Gradient.pvd",
        "adjoint_output": False,
        "adjoint_filename": None,
        "debug_output": False,
    }
    return dictionary


def start_new_annotation():
    set_working_tape(Tape())
    continue_annotation()


def build_wave(automatic_adjoint):
    wave_obj = spyro.AcousticWave(
        dictionary=set_dictionary(automatic_adjoint=automatic_adjoint)
    )
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    return wave_obj


def build_exact_receivers():
    wave_obj_exact = build_wave(automatic_adjoint=False)
    cond = fire.conditional(wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(conditional=cond)
    assert wave_obj_exact.forward_solve() is None
    return wave_obj_exact.receivers_data


def build_true_recv(rec_out_exact, true_recv_format):
    if true_recv_format is None:
        return None
    if true_recv_format == "list":
        return [row.copy() for row in rec_out_exact]
    return rec_out_exact.copy()


def compute_functional(wave_obj_guess, true_recv):
    start_new_annotation()
    previous_compute_functional = wave_obj_guess.compute_functional
    wave_obj_guess.compute_functional = True
    try:
        assert wave_obj_guess.forward_solve(true_recv=true_recv) is None
        return wave_obj_guess.functional
    finally:
        wave_obj_guess.compute_functional = previous_compute_functional


def compute_functional_value(wave_obj_guess, rec_out_exact, solver_case):
    if solver_case["automatic_adjoint"]:
        return compute_functional(
            wave_obj_guess,
            build_true_recv(rec_out_exact, solver_case["true_recv_format"]),
        )

    assert wave_obj_guess.forward_solve() is None
    misfit = rec_out_exact - wave_obj_guess.receivers_data
    return spyro.utils.compute_functional(wave_obj_guess, misfit)


def get_gradient_and_functional(wave_obj_guess, rec_out_exact, solver_case):
    if solver_case["automatic_adjoint"]:
        start_new_annotation()
        dJ = wave_obj_guess.gradient_solve(
            true_recv=build_true_recv(rec_out_exact, solver_case["true_recv_format"])
        )
        Jm = wave_obj_guess.functional
    else:
        assert wave_obj_guess.forward_solve() is None
        misfit = rec_out_exact - wave_obj_guess.receivers_data
        Jm = spyro.utils.compute_functional(wave_obj_guess, misfit)
        dJ = wave_obj_guess.gradient_solve(
            misfit=misfit,
            forward_solution=deepcopy(wave_obj_guess.forward_solution),
        )

    return dJ, Jm


def check_gradient(wave_obj_guess, dJ, Jm, rec_out_exact, solver_case):
    errors = []
    direction = fire.Function(wave_obj_guess.function_space)
    rng = np.random.default_rng(0)
    direction.dat.data[:] = rng.random(direction.dat.data.shape)

    base_velocity = fire.Function(wave_obj_guess.function_space, name="velocity")
    base_velocity.assign(wave_obj_guess.c)

    projnorm = fire.assemble(
        dJ * direction * fire.dx(**wave_obj_guess.quadrature_rule)
    )

    for step in STEPS:
        wave_obj_guess.reset_pressure()
        perturbed_velocity = fire.Function(
            wave_obj_guess.function_space, name="velocity"
        )
        perturbed_velocity.assign(base_velocity)
        perturbed_velocity.dat.data[:] += step * direction.dat.data_ro[:]
        wave_obj_guess.initial_velocity_model = perturbed_velocity

        J_plusdm = compute_functional_value(
            wave_obj_guess, rec_out_exact, solver_case
        )
        grad_fd = (J_plusdm - Jm) / step
        errors.append(abs(100 * ((grad_fd - projnorm) / projnorm)))

    wave_obj_guess.initial_velocity_model = base_velocity

    assert errors[-1] < 1
    assert errors[-1] < errors[0]


@pytest.mark.slow
@pytest.mark.parametrize(
    "solver_case",
    [
        pytest.param(
            {"automatic_adjoint": False, "true_recv_format": None},
            id="implemented",
        ),
        pytest.param(
            {"automatic_adjoint": True, "true_recv_format": "array"},
            id="automatic-array",
        ),
        pytest.param(
            {"automatic_adjoint": True, "true_recv_format": "list"},
            id="automatic-list",
        ),
    ],
)
def test_gradient(solver_case):
    rec_out_exact = build_exact_receivers()
    wave_obj_guess = build_wave(
        automatic_adjoint=solver_case["automatic_adjoint"]
    )
    wave_obj_guess.set_initial_velocity_model(constant=2.0)

    dJ, Jm = get_gradient_and_functional(
        wave_obj_guess, rec_out_exact, solver_case
    )

    assert isinstance(dJ, fire.Function)
    assert wave_obj_guess.receivers_data is not None
    if solver_case["automatic_adjoint"]:
        assert wave_obj_guess.functional is not None

    check_gradient(wave_obj_guess, dJ, Jm, rec_out_exact, solver_case)
