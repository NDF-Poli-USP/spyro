"""Taylor verification for the 3D isotropic elastic automated adjoint.

Verifies the automated-adjoint gradient of the L2 misfit functional with
respect to the three isotropic elastic material parameters: rho, lambda, and
mu.
"""

import numpy as np
import pytest
import firedrake as fire
import spyro
from pyadjoint import AdjFloat, Tape


def make_dictionary(density, lmbda, mu):
    """Build the model dictionary for a small 3D isotropic elastic problem."""
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 3,
            "dimension": 3,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 1.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([0.0, 1.0, 0.0]),
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.3, 0.5), (-0.8, 0.7, 0.5), 3
            ),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 1.0,
            "dt": 0.0005,
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
            "density": density,
            "lambda": lmbda,
            "mu": mu,
            "real_velocity_file": None,
        },
    }


def get_exact_receiver_data():
    """Run the exact 3D forward model and return receiver data."""
    wave_exact = spyro.IsotropicWave(
        make_dictionary(density=0.1, lmbda=0.025, mu=0.1)
    )
    wave_exact.set_mesh(input_mesh_parameters={"edge_length": 0.25, "periodic": True})
    wave_exact.forward_solve()
    return wave_exact.forward_solution_receivers


@pytest.mark.slow
def test_elastic_automated_adjoint_3d():
    """Check the 3D elastic automated adjoint for rho, lambda, and mu."""
    rec_out_exact = get_exact_receiver_data()

    wave_guess = spyro.IsotropicWave(
        make_dictionary(density=0.12, lmbda=0.20, mu=0.08)
    )
    wave_guess.set_mesh(input_mesh_parameters={"edge_length": 0.25, "periodic": True})
    wave_guess.real_shot_record = rec_out_exact
    wave_guess.enable_automated_adjoint()
    wave_guess.forward_solve()

    assert isinstance(wave_guess.automated_adjoint._tape, Tape), (
        "Pyadjoint tape is not a Tape instance after forward solve."
    )
    assert isinstance(wave_guess.functional_value, AdjFloat), (
        f"Expected wave_guess.functional_value to be an AdjFloat, "
        f"got {type(wave_guess.functional_value)}."
    )

    controls = wave_guess.automated_adjoint.controls
    assert len(controls) == 3, (
        f"Expected three elastic controls, got {len(controls)}."
    )

    wave_guess.automated_adjoint.create_reduced_functional(
        wave_guess.functional_value
    )

    rng = np.random.default_rng(43)
    direction = [
        fire.Function(
            control.function_space(),
            val=0.1 * rng.random(control.function_space().dim()),
        )
        for control in controls
    ]
    conv_rate = wave_guess.automated_adjoint.verify_gradient(
        controls,
        direction,
    )
    assert conv_rate > 1.95, (
        f"Taylor test convergence rate {conv_rate:.4f} < 1.95. "
        "The 3D automated adjoint gradient is likely incorrect."
    )

    wave_guess.automated_adjoint.clear_tape()
    assert wave_guess.automated_adjoint._tape is None


if __name__ == "__main__":
    test_elastic_automated_adjoint_3d()
