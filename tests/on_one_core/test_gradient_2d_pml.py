import numpy as np
import firedrake as fire
import pytest
import spyro
from pyadjoint import Tape, continue_annotation, set_working_tape


def check_manual_gradient(wave_obj_guess, dJ, rec_out_exact, Jm):
    step = 1e-3
    direction = fire.Function(wave_obj_guess.function_space)
    direction.assign(dJ)

    wave_obj_guess.reset_pressure()
    wave_obj_guess.initial_velocity_model = fire.Constant(2.0) + step * direction
    assert wave_obj_guess.forward_solve() is None

    misfit_plusdm = rec_out_exact - wave_obj_guess.receivers_data
    J_plusdm = spyro.utils.compute_functional(wave_obj_guess, misfit_plusdm)

    grad_fd = (J_plusdm - Jm) / step
    projnorm = fire.assemble(
        dJ * direction * fire.dx(**wave_obj_guess.quadrature_rule)
    )
    error = np.abs(100 * ((grad_fd - projnorm) / projnorm))

    assert error < 5


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


def build_models(automatic_adjoint):
    dictionary = set_dictionary(automatic_adjoint)

    wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    cond = fire.conditional(wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    assert wave_obj_exact.forward_solve() is None
    rec_out_exact = wave_obj_exact.receivers_data

    if automatic_adjoint:
        set_working_tape(Tape())
        continue_annotation()

    wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    wave_obj_guess.set_initial_velocity_model(constant=2.0)

    return rec_out_exact, wave_obj_guess


@pytest.mark.slow
@pytest.mark.parametrize("automatic_adjoint", [False, True])
def test_gradient_pml(automatic_adjoint):
    rec_out_exact, wave_obj_guess = build_models(automatic_adjoint)

    if automatic_adjoint:
        dJ = wave_obj_guess.gradient_solve(
            true_recv=rec_out_exact,
        )
        assert wave_obj_guess.functional is not None
    else:
        assert wave_obj_guess.forward_solve() is None
        misfit = rec_out_exact - wave_obj_guess.receivers_data
        Jm = spyro.utils.compute_functional(wave_obj_guess, misfit)
        dJ = wave_obj_guess.gradient_solve(misfit=misfit)
        check_manual_gradient(wave_obj_guess, dJ, rec_out_exact, Jm)

    assert wave_obj_guess.receivers_data is not None
    assert dJ is not None
