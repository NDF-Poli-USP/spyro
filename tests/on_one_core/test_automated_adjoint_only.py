"""Adjoint-only reverse sweep of the automated adjoint.

:meth:`AutomatedAdjoint.compute_adjoint` runs the adjoint of the tape without
assembling the derivative with respect to the controls. The adjoint of every
time step must be the one a gradient computation produces.
"""

from collections.abc import Callable

import numpy as np
import pytest
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock
from pyadjoint import get_working_tape

import spyro


def build_wave(velocity: float, free_surface: bool) -> spyro.AcousticWave:
    """Small constant-velocity acoustic problem.

    Parameters
    ----------
    velocity : float
        Velocity (km/s).
    free_surface : bool
        Pressure-free (Dirichlet) top instead of the natural one.

    Returns
    -------
    spyro.AcousticWave
        Wave with mesh and velocity set.
    """
    dictionary = {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2,
                    "dimension": 2},
        "parallelism": {"type": "automatic"},
        "mesh": {"length_z": 1.0, "length_x": 1.0, "length_y": 0.0,
                 "mesh_file": None, "mesh_type": "firedrake_mesh"},
        "acquisition": {"source_type": "ricker",
                        "source_locations": [(-0.1, 0.5)],
                        "frequency": 5.0, "delay": 1.5,
                        "delay_type": "multiples_of_minimum",
                        "receiver_locations": spyro.create_transect(
                            (-0.6, 0.2), (-0.6, 0.8), 4)},
        "absorving_boundary_conditions": {"status": False,
                                          "free_surface": free_surface},
        "time_axis": {"initial_time": 0.0, "final_time": 0.4, "dt": 0.002,
                      "amplitude": 1, "output_frequency": 10**9,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None, "gradient_output": False,
                          "gradient_filename": None, "adjoint_output": False,
                          "adjoint_filename": None, "debug_output": False},
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=velocity)
    return wave


def step_adjoints(sweep: Callable[[], object]) -> dict[int, np.ndarray]:
    """Collect the adjoint of every time-step solve during a reverse sweep.

    Parameters
    ----------
    sweep : callable
        Function without arguments that runs the reverse sweep.

    Returns
    -------
    dict of int to numpy.ndarray
        Adjoint values of each time step, keyed by the step.
    """
    blocks = [block for block in get_working_tape().get_blocks()
              if isinstance(block, GenericSolveBlock)]
    adjoints = {}
    for step, block in enumerate(blocks):
        block.adj_cb = (lambda adjoint, step=step:
                        adjoints.__setitem__(step, adjoint.dat.data_ro.copy()))
    sweep()
    return adjoints


@pytest.mark.parametrize("free_surface", [False, True])
def test_adjoint_only_matches_gradient_sweep(free_surface: bool) -> None:
    """The adjoint-only sweep reproduces every step adjoint of the gradient
    sweep and leaves the control without an adjoint value.

    Parameters
    ----------
    free_surface : bool
        Pressure-free or natural top.
    """
    observed = build_wave(2.0, free_surface)
    observed.forward_solve()
    wave = build_wave(1.8, free_surface)
    wave.real_shot_record = observed.forward_solution_receivers
    wave.enable_automated_adjoint()
    wave.forward_solve()
    driver = wave.automated_adjoint

    only = step_adjoints(lambda: driver.compute_adjoint(wave.functional_value))
    assert all(control.block_variable.adj_value is None
               for control in driver.controls)

    driver.create_reduced_functional(wave.functional_value)
    full = step_adjoints(driver.compute_derivative)

    assert sorted(only) == sorted(full) == list(range(len(full)))
    scale = max(np.abs(values).max() for values in full.values())
    assert scale > 0.0
    for step, values in full.items():
        np.testing.assert_allclose(only[step], values, rtol=1e-12,
                                   atol=1e-12 * scale)
    driver.clear_tape()
