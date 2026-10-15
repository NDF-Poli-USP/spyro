"""Point source written as a scalar amplitude times a fixed spatial term.

Without the PML, the vertex-only-mesh point source enters the time step as
:math:`s(t) \\int q_h v \\, dx`, with :math:`M q_h = q` for the source
cofunction :math:`q`. The term must assemble back to :math:`s(t)\\, q`, and
the pyadjoint tape must then record no source cofunction per time step, only
the scalar amplitude.
"""

import numpy as np
import pytest

import firedrake as fire
import spyro
from spyro.solvers.acoustic_solver_construction_no_pml import set_point_source

from .test_gradient_auto_adjoint_nrbc import (
    EDGE_LENGTH, build_dictionary, observed_record,
)


def guess_wave(absorbing_sides: dict | None,
               use_vertex_only_mesh: bool = False) -> spyro.AcousticWave:
    """Build a guess model with a heterogeneous velocity.

    Parameters
    ----------
    absorbing_sides : dict or None
        ``absorb_<side>`` flags of the non-reflecting boundary condition, or
        ``None`` for reflecting boundaries.
    use_vertex_only_mesh : bool, optional
        Whether sources and receivers use vertex-only meshes. The automated
        adjoint turns this on regardless.

    Returns
    -------
    spyro.AcousticWave
        The wave object, with mesh and velocity model set, not yet solved.
    """
    dictionary = build_dictionary(absorbing_sides)
    dictionary["acquisition"]["use_vertex_only_mesh"] = use_vertex_only_mesh
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(
        conditional=2.0 - 0.5 * wave.mesh_z
        + 0.2 * fire.exp(-((wave.mesh_x - 0.5) ** 2) / 0.02),
        dg_velocity_model=False,
    )
    return wave


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    """Relative error of ``value`` with respect to ``reference``.

    Parameters
    ----------
    value : numpy.ndarray
        Computed array.
    reference : numpy.ndarray
        Reference array, assumed nonzero.

    Returns
    -------
    float
        ``||value - reference|| / ||reference||``.
    """
    return np.linalg.norm(value - reference) / np.linalg.norm(reference)


def test_source_term_assembles_to_point_source() -> None:
    """The amplitude times the source density reproduces the cofunction."""
    wave = guess_wave(None, use_vertex_only_mesh=True)
    wave._initialize_model_parameters()
    wave.matrix_building()
    wave.sources.current_sources = [0]
    source_cofunction = wave.sources.source_cofunction()

    set_point_source(wave, source_cofunction)
    amplitude = 0.37
    wave.source_amplitude.assign(amplitude)
    v = fire.TestFunction(wave.function_space)
    source_term = fire.assemble(
        wave.source_amplitude * wave.source_density * v
        * fire.dx(**wave.quadrature_rule)
    )

    expected = amplitude * source_cofunction.dat.data_ro
    assert np.linalg.norm(expected) > 0
    assert relative_error(source_term.dat.data_ro, expected) < 1e-12


def field_outputs_per_step(wave) -> list[int]:
    """Count the outputs living on the dual space recorded in each time step.

    Parameters
    ----------
    wave : spyro.AcousticWave
        Wave object whose forward solve was recorded on its tape.

    Returns
    -------
    list of int
        For every recorded time step, the number of block outputs that are
        cofunctions of ``wave.function_space``.
    """
    dual = wave.function_space.dual()
    counts = []
    for step in wave.automated_adjoint._tape.timesteps:
        counts.append(sum(
            isinstance(output.output, fire.Cofunction)
            and output.output.function_space() == dual
            for block in step for output in block.get_outputs()
        ))
    return counts


@pytest.mark.newer_firedrake
def test_tape_records_scalar_source_amplitude() -> None:
    """After the first step, no time step records a source cofunction.

    The source density is set up once, before the time loop; each step then
    only assigns the scalar amplitude.
    """
    wave = guess_wave(None)
    wave.real_shot_record = observed_record(None)
    # Checkpointing splits the tape into time steps; this schedule keeps
    # every step.
    wave.enable_automated_adjoint(checkpointing=True)
    try:
        wave.forward_solve()
        wave.automated_adjoint.stop_recording()
        counts = field_outputs_per_step(wave)
        assert len(counts) > 2
        assert counts[1:] == [0] * (len(counts) - 1)
        amplitude_assigned = [
            any(output.output is wave.source_amplitude
                for block in step for output in block.get_outputs())
            for step in wave.automated_adjoint._tape.timesteps
        ]
        assert all(amplitude_assigned)
    finally:
        wave.automated_adjoint.clear_tape()


@pytest.mark.slow
def test_implemented_adjoint_switches_off_point_source() -> None:
    """The implemented adjoint does not see the forward point source.

    The implemented adjoint reuses the forward solver and injects the
    adjoint source through the source cofunction. The forward solve leaves
    the point source amplitude at the last wavelet value, which the backward
    solve must switch off. That value vanishes in a plain run, because the
    wavelet has decayed, so it is set to a nonzero value here.
    """
    record = observed_record(None)
    wave = guess_wave(None, use_vertex_only_mesh=True)
    wave.enable_implemented_adjoint()

    gradients = []
    for amplitude in (0.0, 1.0):
        wave.forward_solve()
        assert fire.norm(wave.source_density) > 0
        receivers = np.array(wave.forward_solution_receivers, copy=True)
        wave.source_amplitude.assign(amplitude)
        gradient = wave.gradient_solve(misfit=record - receivers)
        gradients.append(np.array(gradient.dat.data_ro, copy=True))

    assert np.linalg.norm(gradients[0]) > 0
    assert relative_error(gradients[1], gradients[0]) < 1e-12
