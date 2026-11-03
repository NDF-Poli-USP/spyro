"""Quadratic damping layer of the acoustic solver without PML.

The layer occupies the outer ``pad_length`` of the mesh on the bottom and the
sides; it is zero inside and absorbs what reaches the pad, so the late traces,
which carry the waves coming back from the boundaries, lose energy.
"""

import numpy as np
import pytest

import spyro
from spyro.solvers.acoustic_solver_construction_no_pml import damping_profile

PAD = 0.25


def build_wave(damping: bool) -> spyro.AcousticWave:
    """Small constant-velocity problem with a first-order absorbing boundary.

    Parameters
    ----------
    damping : bool
        Add the damping layer.

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
                            (-0.2, 0.3), (-0.2, 0.7), 5)},
        "absorving_boundary_conditions": {
            "status": True, "abc_type": "nrbc", "free_surface": True,
            "damping_layer": ({"pad_length": PAD, "reflection": 1e-3}
                              if damping else None)},
        "time_axis": {"initial_time": 0.0, "final_time": 1.5, "dt": 0.002,
                      "amplitude": 1, "output_frequency": 10**9,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None, "gradient_output": False,
                          "gradient_filename": None, "adjoint_output": False,
                          "adjoint_filename": None, "debug_output": False},
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    wave.set_initial_velocity_model(constant=1.5)
    return wave


def test_damping_profile_is_zero_inside_and_grows_into_the_pad() -> None:
    """The coefficient vanishes in the interior and reaches its maximum at
    the outer boundary."""
    wave = build_wave(damping=True)
    wave.forward_solve()
    gamma = damping_profile(wave).dat.data_ro
    gamma_max = 3.0 * 1.5 * np.log(1e3) / (2.0 * PAD)
    assert gamma.min() == pytest.approx(0.0)
    assert gamma.max() == pytest.approx(gamma_max, rel=1e-6)
    assert np.count_nonzero(gamma) < gamma.size


def test_damping_layer_absorbs_late_arrivals() -> None:
    """Early traces are unchanged; late traces, made of waves returning from
    the boundaries, lose energy."""
    plain = build_wave(damping=False)
    plain.forward_solve()
    damped = build_wave(damping=True)
    damped.forward_solve()
    a = plain.forward_solution_receivers
    b = damped.forward_solution_receivers
    steps = a.shape[0]
    # Up to 0.15 s nothing has gone to the layer and back to the receivers.
    early = slice(0, steps // 10)
    late = slice(steps // 2, steps)
    assert np.abs(a[early] - b[early]).max() < 1e-3 * np.abs(a).max()
    assert np.sum(b[late] ** 2) < 0.8 * np.sum(a[late] ** 2)
