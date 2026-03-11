from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest
import spyro
from pyadjoint import continue_annotation, pause_annotation
from pyadjoint.tape import get_working_tape


STEPS = (1e-3,)


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    _ = set_test_tape


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
        "Lz": 1.0,
        "Lx": 1.0,
        "Ly": 0.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.1, 0.5)],
        "frequency": 5.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": spyro.create_transect((-0.8, 0.1), (-0.8, 0.9), 10),
        "use_vertex_only_mesh": True,
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": 1.0,
        "dt": 0.0002,
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
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    return dictionary


def start_new_annotation():
    continue_annotation()


def stop_new_annotation():
    pause_annotation()
    tape = get_working_tape()
    if tape is not None:
        tape.clear_tape()


def build_wave(automatic_adjoint):
    wave_obj = spyro.AcousticWave(
        dictionary=set_dictionary(automatic_adjoint=automatic_adjoint)
    )
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    return wave_obj


def assert_forward_solution_length(wave_obj):
    expected_length = sum(
        1
        for step in range(int(wave_obj.final_time / wave_obj.dt) + 1)
        if step % wave_obj.gradient_sampling_frequency == 0
    )
    assert len(wave_obj.forward_solution) == expected_length


def build_exact_receivers():
    wave_obj_exact = build_wave(automatic_adjoint=False)
    cond = fire.conditional(wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    assert wave_obj_exact.forward_solve() is None
    assert_forward_solution_length(wave_obj_exact)
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
        assert_forward_solution_length(wave_obj_guess)
        return float(wave_obj_guess.functional)
    finally:
        wave_obj_guess.compute_functional = previous_compute_functional
        stop_new_annotation()


def compute_functional_value(wave_obj_guess, rec_out_exact, solver_case):
    if solver_case["automatic_adjoint"]:
        return compute_functional(
            wave_obj_guess,
            build_true_recv(rec_out_exact, solver_case["true_recv_format"]),
        )

    assert wave_obj_guess.forward_solve() is None
    misfit = rec_out_exact - wave_obj_guess.receivers_data
    return spyro.utils.compute_functional(wave_obj_guess, misfit)


def apply_pml_gradient_mask(wave_obj, gradient):
    mask = fire.Function(wave_obj.function_space)
    z_min = -wave_obj.mesh_parameters.length_z
    x_min = 0.0
    x_max = wave_obj.mesh_parameters.length_x
    cond = fire.conditional(wave_obj.mesh_z < z_min, 1, 0)
    cond = fire.conditional(wave_obj.mesh_x < x_min, 1, cond)
    cond = fire.conditional(wave_obj.mesh_x > x_max, 1, cond)
    mask.interpolate(cond)

    masked_gradient = fire.Function(wave_obj.function_space, name="gradient")
    masked_gradient.assign(gradient)
    masked_gradient.dat.data[mask.dat.data[:] > 0.95] = 0.0
    return masked_gradient


def get_gradient_and_functional(wave_obj_guess, rec_out_exact, solver_case):
    if solver_case["automatic_adjoint"]:
        start_new_annotation()
        try:
            dJ = wave_obj_guess.gradient_solve(
                true_recv=build_true_recv(
                    rec_out_exact, solver_case["true_recv_format"]
                )
            )
            assert_forward_solution_length(wave_obj_guess)
            Jm = wave_obj_guess.functional
        finally:
            stop_new_annotation()
    else:
        assert wave_obj_guess.forward_solve() is None
        assert_forward_solution_length(wave_obj_guess)
        misfit = rec_out_exact - wave_obj_guess.receivers_data
        Jm = spyro.utils.compute_functional(wave_obj_guess, misfit)
        dJ = wave_obj_guess.gradient_solve(
            misfit=misfit,
            forward_solution=deepcopy(wave_obj_guess.forward_solution),
        )

    return apply_pml_gradient_mask(wave_obj_guess, dJ), Jm


def build_direction(wave_obj_guess, dJ, solver_case):
    direction = fire.Function(wave_obj_guess.function_space, name="direction")
    if solver_case["perturbation_direction"] == "gradient":
        direction.assign(dJ)
        return direction

    rng = np.random.default_rng(0)
    direction.dat.data[:] = rng.random(direction.dat.data.shape)
    return apply_pml_gradient_mask(wave_obj_guess, direction)


def check_gradient(wave_obj_guess, dJ, Jm, rec_out_exact, solver_case):
    errors = []
    direction = build_direction(wave_obj_guess, dJ, solver_case)

    base_velocity = fire.Function(
        wave_obj_guess.function_space, name="velocity"
    )
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

    assert errors[-1] < 5


@pytest.mark.slow
@pytest.mark.parametrize(
    "solver_case",
    [
        pytest.param(
            {
                "automatic_adjoint": False,
                "true_recv_format": None,
                "perturbation_direction": "gradient",
            },
            id="implemented",
        ),
        pytest.param(
            {
                "automatic_adjoint": True,
                "true_recv_format": "array",
                "perturbation_direction": "random",
            },
            id="automatic-array",
        ),
        pytest.param(
            {
                "automatic_adjoint": True,
                "true_recv_format": "list",
                "perturbation_direction": "random",
            },
            id="automatic-list",
        ),
    ],
)
def test_gradient_pml(solver_case):
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
