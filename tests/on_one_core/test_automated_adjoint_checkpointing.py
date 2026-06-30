"""Serial API checks for automated-adjoint checkpointing."""

from copy import deepcopy

import firedrake as fire
import pytest

from checkpoint_schedules import PeriodicDiskRevolve, Revolve

import spyro


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 4,
    "dimension": 2,
}

dictionary["parallelism"] = {
    "type": "spatial",
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
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.5,
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


def get_short_dictionary():
    short_dictionary = deepcopy(dictionary)
    short_dictionary["acquisition"]["receiver_locations"] = spyro.create_transect(
        (-0.8, 0.3), (-0.8, 0.7), 4
    )
    short_dictionary["time_axis"]["final_time"] = 0.05
    short_dictionary["time_axis"]["dt"] = 0.001
    return short_dictionary


@pytest.mark.newer_firedrake
def test_enable_automated_adjoint_checkpointing_requires_schedule():
    wave = spyro.AcousticWave(dictionary=dictionary)
    with pytest.raises(ValueError, match="checkpoint_schedule"):
        wave.enable_automated_adjoint(checkpointing=True)


@pytest.mark.newer_firedrake
def test_checkpointing_start_recording_installs_checkpoint_manager():
    wave = spyro.AcousticWave(dictionary=dictionary)
    minimal_schedule = Revolve(2, 1)
    wave.enable_automated_adjoint(
        checkpointing=True,
        # This is only the smallest finite schedule needed to initialize the
        # manager; production FWI tests should pass a schedule built from nt.
        checkpoint_schedule=minimal_schedule,
    )

    tape = wave.automated_adjoint.start_recording()
    assert tape is wave.automated_adjoint._tape
    assert isinstance(tape._checkpoint_manager, spyro.SpyroCheckpointManager)
    wave.automated_adjoint.clear_tape()


@pytest.mark.newer_firedrake
def test_decimated_recompute_requires_positive_period():
    with pytest.raises(ValueError, match="period"):
        spyro.DecimatedRecompute(period=0)


@pytest.mark.newer_firedrake
def test_checkpointing_start_recording_installs_decimated_recompute_strategy():
    wave = spyro.AcousticWave(dictionary=dictionary)
    schedule = PeriodicDiskRevolve(12, 3, wd=1000, rd=1000)
    strategy = spyro.DecimatedRecompute(period=2)

    wave.enable_automated_adjoint(
        checkpointing=True,
        checkpoint_schedule=schedule,
        checkpoint_recompute_strategy=strategy,
    )

    tape = wave.automated_adjoint.start_recording()
    assert tape._checkpoint_manager.recompute_strategy is strategy
    wave.automated_adjoint.clear_tape()


@pytest.mark.newer_firedrake
def test_decimated_recompute_computes_gradient_with_periodic_disk_revolve():
    short_dictionary = get_short_dictionary()

    exact = spyro.AcousticWave(dictionary=short_dictionary)
    exact.set_mesh(input_mesh_parameters={"edge_length": 0.3})
    exact.set_initial_velocity_model(
        conditional=fire.conditional(exact.mesh_z > -0.5, 1.5, 3.5),
        dg_velocity_model=False,
    )
    exact.forward_solve()

    guess = spyro.AcousticWave(dictionary=short_dictionary)
    guess.real_shot_record = exact.forward_solution_receivers
    guess.set_mesh(input_mesh_parameters={"edge_length": 0.3})
    guess.set_initial_velocity_model(constant=2.0)

    nt = int(guess.final_time / guess.dt) + 1
    schedule = PeriodicDiskRevolve(nt, 3, wd=1000, rd=1000)
    strategy = spyro.DecimatedRecompute(period=2)
    guess.enable_automated_adjoint(
        checkpointing=True,
        checkpoint_schedule=schedule,
        checkpoint_recompute_strategy=strategy,
    )

    guess.forward_solve()
    guess.automated_adjoint.create_reduced_functional(guess.functional_value)
    gradient = guess.automated_adjoint.compute_gradient()

    assert isinstance(gradient, fire.Function)
    assert gradient.dat.data.shape == guess.c.dat.data.shape
    guess.automated_adjoint.clear_tape()
