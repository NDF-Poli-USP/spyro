"""Serial API checks for automated-adjoint checkpointing."""

import pytest

from checkpoint_schedules import Revolve

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
