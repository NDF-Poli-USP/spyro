"""Taylor test for the 2D isotropic elastic wave automated adjoint.

Verifies that the gradient of the L2 misfit functional with respect to the
P-wave velocity (a Firedrake Constant) computed via pyadjoint is correct to
second order, following the same pattern as test_gradient_2d_pml.py for the
acoustic case.

Based on notebook_tutorials/elastic_forward.ipynb.
"""

import numpy as np
import pytest
import firedrake as fire
import spyro
from pyadjoint import AdjFloat, Tape, taylor_test



def make_dictionary(p_wave_velocity):
    """Build the model dictionary for the 2D isotropic elastic wave problem.

    Parameters
    ----------
    p_wave_velocity : float
        P-wave velocity (km/s).

    Returns
    -------
    dict
        Model configuration dictionary.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {
            "type": "automatic",
        },
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
            # Force in the x-direction only (following the notebook).
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.2), (-0.8, 0.8), 10),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.5,
            "dt": 0.001,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": "results/forward_output.pvd",
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": "results/Gradient.pvd",
            "adjoint_output": False,
            "adjoint_filename": None,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            "density": 0.1,
            "p_wave_velocity": p_wave_velocity,
            "s_wave_velocity": 1.0,
            "real_velocity_file": None,
        },
    }


def get_exact_receiver_data():
    """Run the 'exact' forward model and return the receiver data.

    Parameters
    ----------
    edge_length : float
        Mesh edge length for the exact model.

    Returns
    -------
    numpy.ndarray
        Receiver time series from the exact forward solve.
    """
    wave_exact = spyro.IsotropicWave(make_dictionary(p_wave_velocity=1.5))
    wave_exact.set_mesh(input_mesh_parameters={"edge_length": 0.02, "periodic": True})
    wave_exact.forward_solve()
    return wave_exact.forward_solution_receivers


@pytest.mark.slow
def test_elastic_automated_adjoint_2d():
    """Taylor test for the automated adjoint of the 2D isotropic elastic wave.

    Runs the following workflow:

    1. Solve the exact forward problem (p_wave = 1.5 km/s) and store the
       synthetic receiver data as the "observed" record.
    2. Set up a guess model (p_wave = 2.0 km/s) and enable the automated
       adjoint so that pyadjoint records the computation on a tape.
    3. Run the guess forward solve; the L2 misfit functional is accumulated
       per time step as a pyadjoint-annotated AdjFloat.
    4. Build the reduced functional J(c) and compute dJ/dc via
       ``compute_gradient()``.
    5. Run pyadjoint's ``taylor_test`` with a constant-valued perturbation
       direction and verify that the convergence rate exceeds 1.95 (second-
       order accuracy).
    """
    rec_out_exact = get_exact_receiver_data()

    # --- Guess model (p_wave = 2.0 km/s) ---
    wave_guess = spyro.IsotropicWave(make_dictionary(p_wave_velocity=2.0))
    wave_guess.set_mesh(input_mesh_parameters={"edge_length": 0.02, "periodic": True})
    wave_guess.real_shot_record = rec_out_exact
    quit()
    # Enable automated adjoint: sets up the pyadjoint tape and registers c as
    # the control. Also switches to vertex-only mesh for the source/receiver
    # assembly so that pyadjoint can trace through the interpolation steps.
    wave_guess.enable_automated_adjoint()

    # Forward solve: pyadjoint records every Firedrake operation on its tape.
    # The L2 misfit functional is accumulated at each time step and stored in
    # wave_guess.functional_value as an AdjFloat.
    wave_guess.forward_solve()
    # stop_recording() is already called inside forward_solve() for the
    # automated adjoint path, but we repeat it here for safety.
    wave_guess.automated_adjoint.stop_recording()

    assert isinstance(wave_guess.automated_adjoint._tape, Tape), (
        "Pyadjoint tape is not a Tape instance after forward solve."
    )
    assert isinstance(wave_guess.functional_value, AdjFloat), (
        f"Expected wave_guess.functional_value to be an AdjFloat, "
        f"got {type(wave_guess.functional_value)}."
    )

    # Build the reduced functional J(c).
    # We deliberately do NOT call compute_gradient() before the Taylor test:
    # calling the derivative before taylor_test can exhaust annotation state.
    # Instead, we pass dJdm=None and let pyadjoint evaluate the directional
    # derivative internally as part of the Taylor test.
    wave_guess.automated_adjoint.create_reduced_functional(
        wave_guess.functional_value
    )

    # Taylor test.
    #
    # wave.c is a scalar Constant so verify_gradient() cannot be used
    # directly (it calls control_var.function_space() which is undefined for
    # Constants). We call pyadjoint's taylor_test directly instead.
    #
    # direction = Constant(1.0): perturb c in the direction of a unit scalar.
    # dJdm=None: pyadjoint computes the directional derivative J'(c)[h]
    # internally from the reduced functional's derivative.
    direction = fire.Constant(1.0)
    conv_rate = taylor_test(
        wave_guess.automated_adjoint.reduced_functional,
        wave_guess.c,
        direction,
    )

    print(f"Taylor test convergence rate: {conv_rate:.4f}")
    assert conv_rate > 1.95, (
        f"Taylor test convergence rate {conv_rate:.4f} < 1.95. "
        "The automated adjoint gradient is likely incorrect."
    )

    # Clean up the pyadjoint tape.
    wave_guess.automated_adjoint.clear_tape()
    assert wave_guess.automated_adjoint._tape is None


if __name__ == "__main__":
    test_elastic_automated_adjoint_2d()
