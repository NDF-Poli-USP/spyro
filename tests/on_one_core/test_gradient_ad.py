import firedrake as fire
import numpy as np
import pytest
import spyro

from spyro.solvers import acoustic_wave as acoustic_wave_module


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    _ = set_test_tape


def set_dictionary():
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 1,
        "dimension": 2,
        "automatic_adjoint": True,
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
        "receiver_locations": spyro.create_transect(
            (-0.8, 0.1), (-0.8, 0.9), 8
        ),
        "use_vertex_only_mesh": True,
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": 0.6,
        "dt": 0.001,
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


def build_exact_receivers():
    dictionary = set_dictionary()
    wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.04})
    cond = fire.conditional(wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    assert wave_obj_exact.forward_solve() is None
    return wave_obj_exact.receivers_data


@pytest.mark.slow
@pytest.mark.parametrize("true_recv_format", ["array", "list"])
def test_gradient_ad_uses_automated_adjoint(monkeypatch, true_recv_format):
    rec_out_exact = build_exact_receivers()
    true_recv = rec_out_exact
    if true_recv_format == "list":
        true_recv = [row.copy() for row in rec_out_exact]

    dictionary = set_dictionary()
    wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.04})
    wave_obj_guess.set_initial_velocity_model(constant=2.0)

    calls = {"count": 0}
    original_compute_gradient = (
        acoustic_wave_module.AutomatedAdjoint.compute_gradient
    )

    def wrapped_compute_gradient(self):
        calls["count"] += 1
        return original_compute_gradient(self)

    monkeypatch.setattr(
        acoustic_wave_module.AutomatedAdjoint,
        "compute_gradient",
        wrapped_compute_gradient,
    )

    assert wave_obj_guess.automatic_adjoint is True
    gradient = wave_obj_guess.gradient_solve(true_recv=true_recv)
    assert calls["count"] == 1
    assert wave_obj_guess.receivers_data is not None
    assert wave_obj_guess.functional is not None
    assert isinstance(gradient, fire.Function)
    assert isinstance(
        wave_obj_guess.automated_adjoint,
        spyro.solvers.AutomatedAdjoint,
    )
    assert wave_obj_guess.automated_adjoint.reduced_functional is not None
    assert wave_obj_guess.automated_adjoint.verify_gradient() > 0.9


@pytest.mark.slow
def test_gradient_ad_repeated_calls_reuse_reduced_functional(monkeypatch):
    rec_out_exact = build_exact_receivers()

    dictionary = set_dictionary()
    wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.04})
    wave_obj_guess.set_initial_velocity_model(constant=2.0)

    calls = {"forward": 0, "recompute": 0}
    original_forward_solve = acoustic_wave_module.AcousticWave.forward_solve
    original_recompute = (
        acoustic_wave_module.AutomatedAdjoint.recompute_functional
    )

    def wrapped_forward_solve(self, **kwargs):
        calls["forward"] += 1
        return original_forward_solve(self, **kwargs)

    def wrapped_recompute(self, control_value):
        calls["recompute"] += 1
        return original_recompute(self, control_value)

    monkeypatch.setattr(
        acoustic_wave_module.AcousticWave,
        "forward_solve",
        wrapped_forward_solve,
    )
    monkeypatch.setattr(
        acoustic_wave_module.AutomatedAdjoint,
        "recompute_functional",
        wrapped_recompute,
    )

    gradient_first = wave_obj_guess.gradient_solve(true_recv=rec_out_exact)
    automated_adjoint = wave_obj_guess.automated_adjoint
    gradient_second = wave_obj_guess.gradient_solve(
        true_recv=[row.copy() for row in rec_out_exact]
    )

    assert calls["forward"] == 1
    assert calls["recompute"] == 1
    assert wave_obj_guess.automated_adjoint is automated_adjoint
    assert wave_obj_guess.automated_adjoint.reduced_functional is not None
    assert np.allclose(
        gradient_first.dat.data_ro,
        gradient_second.dat.data_ro,
        rtol=1e-9,
        atol=1e-9,
    )
    assert wave_obj_guess.automated_adjoint.verify_gradient() > 0.9


def test_gradient_ad_validates_true_receiver_timestep_count():
    rec_out_exact = build_exact_receivers()

    dictionary = set_dictionary()
    wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.04})
    wave_obj_guess.set_initial_velocity_model(constant=2.0)

    with pytest.raises(ValueError, match="unexpected number of timesteps"):
        wave_obj_guess.gradient_solve(true_recv=rec_out_exact[:-1])
