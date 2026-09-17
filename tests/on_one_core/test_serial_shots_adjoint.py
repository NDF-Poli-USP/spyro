"""Automated adjoint over serially propagated shots.

With ``parallelism`` set to ``spatial`` and more than one source, a single
process propagates the shots one after another. The functional sums over
them, so every shot has to be recorded on the same tape for the gradient to
be the gradient of that sum.
"""

import numpy as np
from pyadjoint import Tape
import pytest

import spyro


def build_wave(source_locations):
    """Build a small acoustic wave propagating ``source_locations`` serially.

    Parameters
    ----------
    source_locations : list of tuple
        Source positions. More than one makes the propagation serial.

    Returns
    -------
    spyro.AcousticWave
        Wave with a mesh and a constant velocity model, ready to solve.
    """
    dictionary = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
            "dimension": 2,
        },
        "parallelism": {"type": "spatial"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.2), (-0.8, 0.8), 5,
            ),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.4,
            "dt": 0.002,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }
    wave = spyro.AcousticWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    return wave


def record(source_locations):
    """Run one annotated forward solve and return its tape.

    Parameters
    ----------
    source_locations : list of tuple
        Source positions to propagate.

    Returns
    -------
    pyadjoint.Tape
        The tape the forward solve recorded.
    """
    wave = build_wave(source_locations)
    wave.enable_automated_adjoint()
    steps = int(wave.final_time / wave.dt) + 1
    wave.real_shot_record = np.zeros((steps, 5))
    try:
        wave.forward_solve()
        assert isinstance(wave.automated_adjoint._tape, Tape)
        return wave.automated_adjoint._tape
    finally:
        wave.automated_adjoint.clear_tape()
        # Serial propagation writes one shot file per source; remove them on
        # the failing path too, so a failure does not litter the tree.
        spyro.io.delete_tmp_files(wave)


@pytest.mark.newer_firedrake
@pytest.mark.slow
def test_serial_shots_accumulate_on_one_tape():
    """Every serially propagated shot is recorded, not just the last one."""
    one_shot = len(record([(-0.1, 0.5)]).get_blocks())
    two_shots = len(record([(-0.1, 0.3), (-0.1, 0.7)]).get_blocks())

    # Both shots are on the tape, so it holds twice the operations. Recording
    # only the last would leave it the size of a single shot, and the
    # gradient would silently be that shot's rather than the sum's.
    assert one_shot > 0
    assert two_shots == 2 * one_shot


@pytest.mark.newer_firedrake
@pytest.mark.slow
def test_a_second_forward_solve_starts_its_own_tape():
    """One forward solve is one recording, never appended to an earlier one."""
    wave = build_wave([(-0.1, 0.5)])
    wave.enable_automated_adjoint()
    steps = int(wave.final_time / wave.dt) + 1
    wave.real_shot_record = np.zeros((steps, 5))

    try:
        wave.forward_solve()
        first = len(wave.automated_adjoint._tape.get_blocks())
        wave.forward_solve()
        second = len(wave.automated_adjoint._tape.get_blocks())

        # Same size, not twice it: the second solve replaced the recording
        # rather than adding to it. Otherwise a gradient taken afterwards
        # would be differentiating both solves at once.
        assert first > 0
        assert second == first
    finally:
        wave.automated_adjoint.clear_tape()
        spyro.io.delete_tmp_files(wave)


if __name__ == "__main__":
    pytest.main([__file__])
