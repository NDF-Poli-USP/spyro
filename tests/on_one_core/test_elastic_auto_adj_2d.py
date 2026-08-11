"""Taylor test for the 2D isotropic elastic wave automated adjoint.

Verifies that the gradient of the L2 misfit functional is correct to second
order for both supported isotropic elastic parameterizations: (rho, lambda,
mu) and (rho, P-wave velocity, S-wave velocity).

Based on notebook_tutorials/elastic_forward.ipynb.
"""

import numpy as np
import pytest
import firedrake as fire
import spyro
from pyadjoint import AdjFloat, Tape


def make_dictionary(material_parameters):
    """Build the model dictionary for the 2D isotropic elastic wave problem.

    Parameters
    ----------
    material_parameters : dict
        Either density with the two Lame parameters, or density with P- and
        S-wave velocities.

    Returns
    -------
    dict
        Model configuration dictionary.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 4,
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
            "final_time": 1.0,
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
            **material_parameters,
            "real_velocity_file": None,
        },
    }


def get_exact_receiver_data():
    """Run the 'exact' forward model and return the receiver data.

    Returns
    -------
    numpy.ndarray
        Receiver time series from the exact forward solve.
    """
    wave_exact = spyro.IsotropicWave(
        make_dictionary({"density": 0.1, "lambda": 0.025, "mu": 0.1})
    )
    wave_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1, "periodic": True})
    wave_exact.forward_solve()
    return wave_exact.forward_solution_receivers


@pytest.fixture(scope="module")
def exact_receiver_data():
    """Reuse the observed solve across both control parameterizations."""
    return get_exact_receiver_data()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("guess_material", "control_attributes"),
    [
        pytest.param(
            {"density": 0.12, "lambda": 0.20, "mu": 0.08},
            ("rho", "lmbda", "mu"),
            id="lame",
        ),
        pytest.param(
            {
                "density": 0.12,
                "p_wave_velocity": np.sqrt(3.0),
                "s_wave_velocity": np.sqrt(2.0 / 3.0),
            },
            ("rho", "c", "c_s"),
            id="velocity",
        ),
    ],
)
def test_elastic_automated_adjoint_2d(
    exact_receiver_data,
    guess_material,
    control_attributes,
):
    """Taylor test for the automated adjoint of the 2D isotropic elastic wave.

    Runs the following workflow:

    1. Solve the exact forward problem and store the
       synthetic receiver data as the "observed" record.
    2. Set up a guess model and enable the automated
       adjoint so that pyadjoint records the computation on a tape.
    3. Run the guess forward solve; the L2 misfit functional is accumulated
       per time step as a pyadjoint-annotated AdjFloat.
    4. Build the reduced functional for the active material controls.
    5. Verify the automated-adjoint gradient with a perturbation direction and
       check that the convergence rate exceeds 1.95 (second-order accuracy).
    """
    # --- Guess model ---
    wave_guess = spyro.IsotropicWave(
        make_dictionary(guess_material)
    )
    wave_guess.set_mesh(input_mesh_parameters={"edge_length": 0.05, "periodic": True})
    wave_guess.real_shot_record = exact_receiver_data
    # Enable automated adjoint and register the controls from the active
    # material parameterization. This also switches to the vertex-only mesh so
    # pyadjoint can trace the source/receiver interpolation steps.
    wave_guess.enable_automated_adjoint()

    # Forward solve: pyadjoint records every Firedrake operation on its tape.
    # The L2 misfit functional is accumulated at each time step and stored in
    # wave_guess.functional_value as an AdjFloat.
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
    assert all(
        control is getattr(wave_guess, attribute)
        for control, attribute in zip(controls, control_attributes)
    ), "Automated-adjoint controls do not match the active parameterization."

    wave_guess.automated_adjoint.create_reduced_functional(
        wave_guess.functional_value
    )

    # fixed random seed for reproducibility
    rng = np.random.default_rng(42)
    direction = [
        fire.Function(
            control.function_space(),
            val=0.01 * rng.random(control.function_space().dim()),
        )
        for control in controls
    ]
    conv_rate = wave_guess.automated_adjoint.verify_gradient(
        controls,
        direction,
    )
    assert conv_rate > 1.95, (
        f"Taylor test convergence rate {conv_rate:.4f} < 1.95. "
        "The automated adjoint gradient is likely incorrect."
    )

    # Clean up the pyadjoint tape.
    wave_guess.automated_adjoint.clear_tape()
    assert wave_guess.automated_adjoint._tape is None


if __name__ == "__main__":
    pytest.main([__file__])
