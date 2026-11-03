"""Pressure-free top surface of the acoustic solver.

With ``free_surface`` the top boundary carries the Dirichlet condition
``u = 0`` of a pressure-free surface; without it the top is natural (zero
normal derivative), a rigid surface.
"""

import firedrake as fire
import numpy as np
import pytest

import spyro


def build_wave(free_surface: bool, absorb_top: bool = False) -> spyro.AcousticWave:
    """Small constant-velocity acoustic problem with a source near the top.

    Parameters
    ----------
    free_surface : bool
        Pressure-free top surface.
    absorb_top : bool, optional
        Absorbing top boundary. Default ``False``.

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
                            (-0.3, 0.2), (-0.3, 0.8), 4)},
        "absorving_boundary_conditions": {"status": False,
                                          "free_surface": free_surface,
                                          "absorb_top": absorb_top},
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
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    wave.set_initial_velocity_model(constant=1.5)
    return wave


def top_values(wave: spyro.AcousticWave) -> np.ndarray:
    """Final pressure on the nodes of the top boundary.

    Parameters
    ----------
    wave : spyro.AcousticWave
        A solved wave.

    Returns
    -------
    numpy.ndarray
        Values of the final state on the top-boundary nodes.
    """
    nodes = fire.DirichletBC(wave.function_space, 0.0, 1).nodes
    return wave.u_n.dat.data_ro[nodes]


def test_free_surface_vanishes_on_top() -> None:
    """The free surface keeps the top at zero; the rigid top does not, and
    the two give different traces."""
    free = build_wave(free_surface=True)
    free.forward_solve()
    rigid = build_wave(free_surface=False)
    rigid.forward_solve()
    assert np.all(top_values(free) == 0.0)
    assert np.abs(top_values(rigid)).max() > 0.0
    assert not np.allclose(free.forward_solution_receivers,
                           rigid.forward_solution_receivers)


def test_free_surface_rejects_absorbing_top() -> None:
    """The top cannot be both absorbing and a free surface."""
    with pytest.raises(ValueError):
        build_wave(free_surface=True, absorb_top=True)
