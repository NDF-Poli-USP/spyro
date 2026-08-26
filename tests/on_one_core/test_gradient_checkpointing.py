"""Serial gradient tests for the checkpointed automated adjoint.

Spyro drives the tape with two :mod:`checkpoint_schedules` schedules, chosen
from the snapshot budget given to
:meth:`~spyro.solvers.wave.Wave.enable_automated_adjoint`:

``checkpointing=False``
    No checkpoint manager; the tape keeps every step. This is the reference.
``checkpointing=True, snapshots=None``
    ``SingleMemoryStorageSchedule`` - every adjoint dependency stays in RAM.
``checkpointing=True, snapshots=N``
    ``MixedCheckpointSchedule`` - ``N`` checkpoints in RAM, forward recomputed
    in between.

A schedule changes how the tape is stored, not what it computes, so all three
must produce the same gradient and pass the same Taylor test.
"""

import numpy as np
import pytest

import firedrake as fire
import spyro
from spyro.utils.typing import AdjointType


# Deliberately small: the gradient comparison is exact, so it does not need a
# well-resolved model - only enough time for the source to reach the receivers.
FINAL_TIME = 0.8
EDGE_LENGTH = 0.2
DT = 0.004

# name -> enable_automated_adjoint kwargs
SCHEDULES = {
    "none": {"checkpointing": False, "snapshots": None},
    "single_memory": {"checkpointing": True, "snapshots": None},
    "mixed": {"checkpointing": True, "snapshots": 50},
}


def build_dictionary() -> dict:
    """Build the model dictionary shared by every test in this module.

    Returns
    -------
    dict
        A spyro model dictionary describing a small 2D acoustic problem.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.1), (-0.8, 0.9), 5
            ),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": FINAL_TIME,
            "dt": DT,
            "amplitude": 1,
            "output_frequency": 100000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": None,
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": None,
            "adjoint_output": False,
            "adjoint_filename": None,
            "debug_output": False,
        },
    }


@pytest.fixture(scope="module")
def observed_record():
    """Run the two-layer 'exact' model once and return its receiver data.

    Module scoped: every test compares against the same shot record, and the
    forward solve is the expensive part.

    Returns
    -------
    numpy.ndarray
        Receiver time series used as the real shot record.
    """
    wave = spyro.AcousticWave(dictionary=build_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.5),
        dg_velocity_model=False,
    )
    wave.forward_solve()
    return wave.forward_solution_receivers


def taped_guess(observed_record, schedule: str) -> spyro.AcousticWave:
    """Record a guess-model forward solve under the named schedule.

    Parameters
    ----------
    observed_record : numpy.ndarray
        Receiver data used as the real shot record.
    schedule : str
        Key into :data:`SCHEDULES` selecting the checkpointing settings.

    Returns
    -------
    spyro.AcousticWave
        The wave object, with the tape recorded and annotation stopped.
    """
    wave = spyro.AcousticWave(dictionary=build_dictionary())
    wave.real_shot_record = observed_record
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(constant=2.0)
    wave.enable_automated_adjoint(**SCHEDULES[schedule])
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    return wave


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_taylor_test(observed_record, schedule: str) -> None:
    """Check that every schedule reaches second-order Taylor convergence.

    Parameters
    ----------
    observed_record : numpy.ndarray
        Receiver data used as the real shot record.
    schedule : str
        Key into :data:`SCHEDULES` selecting the checkpointing settings.
    """
    wave = taped_guess(observed_record, schedule)
    wave.automated_adjoint.create_reduced_functional(wave.functional_value)

    size, = np.shape(wave.c.dat.data[:])
    direction = fire.Function(
        wave.c.function_space(), val=np.random.default_rng(0).random(size)
    )
    rate = wave.automated_adjoint.verify_gradient(wave.c, direction=direction)
    assert rate > 1.9, (
        f"Taylor convergence rate {rate} with schedule '{schedule}'"
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_gradient_matches_across_schedules(observed_record) -> None:
    """Check that the three schedules return the same gradient, to round-off.

    A schedule changes how the tape is stored, not what it computes, so this
    is a sharper check than the Taylor test: it catches an adjoint that is
    wrong by a few percent, which a convergence rate would not.

    Parameters
    ----------
    observed_record : numpy.ndarray
        Receiver data used as the real shot record.
    """
    gradients = {}
    for schedule in SCHEDULES:
        wave = taped_guess(observed_record, schedule)
        dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
        gradients[schedule] = np.array(dJ.dat.data_ro, copy=True)
        wave.automated_adjoint.clear_tape()

    reference = gradients["none"]
    assert np.linalg.norm(reference) > 0.0, "reference gradient is zero"

    errors = {
        name: np.linalg.norm(g - reference) / np.linalg.norm(reference)
        for name, g in gradients.items()
    }
    assert all(error < 1e-12 for error in errors.values()), (
        f"Gradients differ between schedules: {errors}"
    )
