import math
from copy import deepcopy

from pyadjoint import annotate_tape

import firedrake as fire
import numpy as np
import pytest
import spyro


def check_gradient(wave_obj_guess, dJ):
    steps = [1e-3, 1e-4, 1e-5]

    errors = []
    remainders = []
    dm = fire.Function(wave_obj_guess.function_space)
    rng = np.random.default_rng(0)
    dm.dat.data[:] = rng.random(dm.dat.data.size)
    Jm = wave_obj_guess.functional_value

    for step in steps:
        wave_obj_guess.reset_pressure()
        wave_obj_guess.initial_velocity_model = fire.Constant(2.0) + step * dm
        wave_obj_guess.forward_solve()
        J_plusdm = wave_obj_guess.functional_value

        grad_fd = (J_plusdm - Jm) / step
        projnorm = fire.assemble(
            dJ * dm * fire.dx(**wave_obj_guess.quadrature_rule)
        )

        errors.append(np.abs(100 * ((grad_fd - projnorm) / projnorm)))
        remainders.append(np.abs(J_plusdm - Jm - step * projnorm))

    errors = np.array(errors)
    remainders = np.array(remainders)

    assert np.all(errors < 1)
    assert np.all(remainders[1:] < 0.2 * remainders[:-1])


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 4,
    "dimension": 2,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, 0.5)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": final_time,
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


def get_real_shot_record():
    wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    cond = fire.conditional(wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(conditional=cond)
    wave_obj_exact.forward_solve()
    return wave_obj_exact.forward_solution_receivers


def get_forward_model(automated_adjoint, real_shot_record=None):
    if real_shot_record is None:
        real_shot_record = get_real_shot_record()

    wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave_obj_guess.set_initial_velocity_model(constant=2.0)
    wave_obj_guess.real_shot_record = real_shot_record

    if automated_adjoint:
        wave_obj_guess.enable_automated_adjoint()
        assert wave_obj_guess.store_forward_time_steps is False
    else:
        wave_obj_guess.enable_implemented_adjoint()
        assert wave_obj_guess._store_misfit is True
        wave_obj_guess.forward_solve()

    return wave_obj_guess


def get_functional_value(automated_adjoint, real_shot_record):
    wave_obj_guess = get_forward_model(
        automated_adjoint,
        real_shot_record=real_shot_record,
    )
    if automated_adjoint:
        with wave_obj_guess.automated_adjoint.fresh_tape():
            wave_obj_guess.forward_solve()
        wave_obj_guess.automated_adjoint.clear_tape()
    return wave_obj_guess.functional_value


def test_automated_functional_matches_manual_value():
    real_shot_record = get_real_shot_record()
    wave_obj_guess = get_forward_model(True, real_shot_record=real_shot_record)

    with wave_obj_guess.automated_adjoint.fresh_tape():
        wave_obj_guess.forward_solve()

    automated_value = wave_obj_guess.functional_value
    manual_value = 0.0
    nt = len(wave_obj_guess.receivers_output)
    for step, simulated_step in enumerate(wave_obj_guess.receivers_output):
        weight = 0.5 if step == 0 or step == nt - 1 else 1.0
        residual_step = np.asarray(real_shot_record[step]) - np.asarray(simulated_step)
        manual_value += (
            0.5 * wave_obj_guess.dt * weight * np.sum(residual_step**2)
        )
    wave_obj_guess.automated_adjoint.clear_tape()

    assert math.isclose(
        automated_value,
        manual_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("automated_adjoint", [False, True])
def test_gradient(automated_adjoint):
    wave_obj_guess = get_forward_model(automated_adjoint)

    if automated_adjoint:
        assert annotate_tape() is False
        dJ = wave_obj_guess.gradient_solve()
        assert isinstance(dJ, fire.Function)

        direction = fire.Function(wave_obj_guess.function_space)
        rng = np.random.default_rng(1)
        direction.dat.data[:] = rng.random(direction.dat.data.size)

        rate = wave_obj_guess.automated_adjoint.verify_gradient(
            wave_obj_guess.c,
            direction=direction,
        )
        assert math.isclose(rate, 2.0, rel_tol=1e-2)
    else:
        assert isinstance(wave_obj_guess.misfit, list)
        forward_solution_guess = deepcopy(wave_obj_guess.forward_solution)
        dJ = wave_obj_guess.gradient_solve(
            forward_solution=forward_solution_guess
        )
        check_gradient(wave_obj_guess, dJ)
