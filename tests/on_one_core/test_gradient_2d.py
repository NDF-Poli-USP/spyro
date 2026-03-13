from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest
import spyro


STEPS = (1e-3, 1e-4, 1e-5)


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


def get_solver_case_id(solver_case):
    if not solver_case["automatic_adjoint"]:
        return "implemented"
    return f'automatic-{solver_case["true_recv_format"]}'


def plot_gradient_errors(errors, steps, solver_case):
    import matplotlib.pyplot as plt

    theory = [errors[0] * step / steps[0] for step in steps]
    case_id = get_solver_case_id(solver_case)
    output = f"gradient_error_verification_{case_id}.png"

    plt.close()
    plt.plot(steps, errors, "o-", label="Error")
    plt.plot(steps, theory, "--", label="first order")
    plt.legend()
    plt.title("Adjoint gradient versus finite difference gradient")
    plt.xlabel("Step")
    plt.ylabel("Error %")
    plt.savefig(output)
    plt.close()
    print(f"Saved gradient error plot to {output}")


def build_wave(automatic_adjoint):
    wave_obj = spyro.AcousticWave(
        dictionary=set_dictionary(automatic_adjoint=automatic_adjoint)
    )
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.1})
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
    cond = fire.conditional(wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(conditional=cond)
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
    control = wave_obj_guess.c
    if control is None:
        control = wave_obj_guess.initial_velocity_model
    automated_adjoint = spyro.solvers.AutomatedAdjoint(control)
    previous_compute_functional = wave_obj_guess.compute_functional
    wave_obj_guess.compute_functional = True
    try:
        with automated_adjoint.fresh_tape():
            automated_adjoint.start_recording()
            try:
                assert wave_obj_guess.forward_solve(true_recv=true_recv) is None
                assert_forward_solution_length(wave_obj_guess)
            finally:
                automated_adjoint.stop_recording()
    finally:
        wave_obj_guess.compute_functional = previous_compute_functional
    return float(wave_obj_guess.functional)


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
        dJ = wave_obj_guess.gradient_solve(
            true_recv=build_true_recv(
                rec_out_exact, solver_case["true_recv_format"]
            )
        )
        assert_forward_solution_length(wave_obj_guess)
        Jm = wave_obj_guess.functional
    else:
        assert wave_obj_guess.forward_solve() is None
        assert_forward_solution_length(wave_obj_guess)
        misfit = rec_out_exact - wave_obj_guess.receivers_data
        Jm = spyro.utils.compute_functional(wave_obj_guess, misfit)
        dJ = wave_obj_guess.gradient_solve(
            misfit=misfit,
            forward_solution=deepcopy(wave_obj_guess.forward_solution),
        )

    return dJ, Jm


def check_gradient(
    wave_obj_guess, dJ, Jm, rec_out_exact, solver_case, plot=False
):
    errors = []
    direction = fire.Function(wave_obj_guess.function_space)
    rng = np.random.default_rng(0)
    direction.dat.data[:] = rng.random(direction.dat.data.shape)

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
        error = abs(100 * ((grad_fd - projnorm) / projnorm))
        errors.append(float(error))

    if plot:
        plot_gradient_errors(errors, STEPS, solver_case)

    wave_obj_guess.initial_velocity_model = base_velocity

    assert errors[-1] < 1
    assert errors[-1] < errors[0]


def run_gradient_case(solver_case, plot=False):
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

    check_gradient(
        wave_obj_guess, dJ, Jm, rec_out_exact, solver_case, plot=plot
    )


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
    run_gradient_case(solver_case)


if __name__ == "__main__":
    run_gradient_case(
        {"automatic_adjoint": True, "true_recv_format": "array"},
        plot=True,
    )
