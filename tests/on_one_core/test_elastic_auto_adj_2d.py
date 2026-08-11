"""Taylor test for the 2D isotropic elastic wave automated adjoint.

Verifies that the gradient of the L2 misfit functional is correct to second
order for both supported isotropic elastic parameterizations and for the
supported natural, essential, and local absorbing boundary formulations.

Based on notebook_tutorials/elastic_forward.ipynb.
"""

import numpy as np
import pytest
import firedrake as fire
import spyro
from pyadjoint import AdjFloat, Tape


pytestmark = pytest.mark.newer_firedrake


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
            "final_time": 0.8,
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


def run_taylor_test(
    dictionary,
    mesh_parameters,
    real_shot_record,
    control_attributes,
    *,
    seed,
    direction_scale=0.01,
    minimum_rate=1.9,
):
    """Record a forward solve and verify all active material gradients."""
    wave_guess = spyro.IsotropicWave(dictionary)
    wave_guess.set_mesh(input_mesh_parameters=mesh_parameters)
    wave_guess.real_shot_record = real_shot_record
    wave_guess.enable_automated_adjoint()

    try:
        wave_guess.forward_solve()

        assert isinstance(wave_guess.automated_adjoint._tape, Tape), (
            "Pyadjoint tape is not a Tape instance after forward solve."
        )
        assert isinstance(wave_guess.functional_value, AdjFloat), (
            f"Expected wave_guess.functional_value to be an AdjFloat, "
            f"got {type(wave_guess.functional_value)}."
        )

        controls = wave_guess.automated_adjoint.controls
        assert len(controls) == len(control_attributes), (
            f"Expected {len(control_attributes)} elastic controls, "
            f"got {len(controls)}."
        )
        assert all(
            control is getattr(wave_guess, attribute)
            for control, attribute in zip(controls, control_attributes)
        ), "Automated-adjoint controls do not match the active parameterization."

        wave_guess.automated_adjoint.create_reduced_functional(
            wave_guess.functional_value
        )
        gradients = wave_guess.automated_adjoint.compute_gradient()
        assert len(gradients) == len(controls)
        assert all(isinstance(gradient, fire.Function) for gradient in gradients)

        rng = np.random.default_rng(seed)
        direction = [
            fire.Function(
                control.function_space(),
                val=(
                    direction_scale
                    * rng.random(control.function_space().dim())
                ),
            )
            for control in controls
        ]
        convergence_rate = wave_guess.automated_adjoint.verify_gradient(
            controls,
            direction,
            dJdm=gradients,
        )
        assert convergence_rate > minimum_rate, (
            "Taylor test convergence rate %.4f < %.2f. The automated "
            "adjoint gradient is likely incorrect."
            % (convergence_rate, minimum_rate)
        )
    finally:
        wave_guess.automated_adjoint.clear_tape()

    assert wave_guess.automated_adjoint._tape is None


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
        pytest.param(
            {
                "density": 0.12,
                "lambda": 0.20,
                "mu": 0.08,
                "control_parameters": ["c_s"],
            },
            ("c_s",),
            id="only-cs-from-lame",
        ),
        pytest.param(
            {
                "density": 0.12,
                "p_wave_velocity": np.sqrt(3.0),
                "s_wave_velocity": np.sqrt(2.0 / 3.0),
                "control_parameters": ["lambda", "mu"],
            },
            ("lmbda", "mu"),
            id="lambda-mu-from-velocity",
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
    run_taylor_test(
        make_dictionary(guess_material),
        {"edge_length": 0.05, "periodic": True},
        exact_receiver_data,
        control_attributes,
        seed=42,
        minimum_rate=1.95,
    )


LAME_GUESS = {"density": 0.12, "lambda": 0.20, "mu": 0.08}
VELOCITY_GUESS = {
    "density": 0.12,
    "p_wave_velocity": np.sqrt(3.0),
    "s_wave_velocity": np.sqrt(2.0 / 3.0),
}


def nrbc_settings(local_abc, time_scheme):
    """Build the input dictionary section for an elastic local ABC."""
    return {
        "absorving_boundary_conditions": {
            "status": True,
            "abc_type": "nrbc",
            "nrbc": {
                "type": local_abc,
                "dt_scheme": time_scheme,
            },
        },
    }


BOUNDARY_CASES = [
    pytest.param(
        LAME_GUESS,
        {},
        ("rho", "lmbda", "mu"),
        0.002,
        id="natural-traction-free",
    ),
    pytest.param(
        LAME_GUESS,
        {
            "boundary_conditions": [
                ("u", "on_boundary", fire.Constant((0.0, 0.0)))
            ]
        },
        ("rho", "lmbda", "mu"),
        0.002,
        id="homogeneous-dirichlet",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "backward"),
        ("rho", "c", "c_s"),
        0.002,
        id="stacey-backward",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("CE_A1", "backward"),
        ("rho", "c", "c_s"),
        0.002,
        id="clayton-engquist-a1-backward",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "central"),
        ("rho", "c", "c_s"),
        0.002,
        id="stacey-central",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "backward_2nd"),
        ("rho", "c", "c_s"),
        0.001,
        id="stacey-backward-2nd",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "guess_material",
        "boundary_settings",
        "control_attributes",
        "time_step",
    ),
    BOUNDARY_CASES,
)
def test_elastic_automated_adjoint_boundary_conditions(
    guess_material,
    boundary_settings,
    control_attributes,
    time_step,
):
    """Taylor-test supported elastic boundary formulations and time schemes."""
    dictionary = make_dictionary(guess_material)
    dictionary.update(boundary_settings)
    dictionary["acquisition"].update({
        "frequency": 10.0,
        "receiver_locations": spyro.create_transect(
            (-0.2, 0.25),
            (-0.2, 0.75),
            5,
        ),
    })
    dictionary["time_axis"].update({
        "final_time": 0.8,
        # The mesh is twice as coarse as the periodic case, so this preserves
        # its CFL ratio while reducing the cost of most Taylor replays. The
        # second-order backward ABC needs the smaller step for stability.
        "dt": time_step,
    })

    number_of_steps = int(
        dictionary["time_axis"]["final_time"]
        / dictionary["time_axis"]["dt"]
    ) + 1
    # A zero target still exercises the complete reduced-functional gradient,
    # while avoiding a second forward solve for every boundary formulation.
    observed_data = np.zeros((number_of_steps, 5, 2))

    run_taylor_test(
        dictionary,
        {"edge_length": 0.1, "periodic": False},
        observed_data,
        control_attributes,
        seed=84,
        # Keep the first perturbation inside the quadratic regime over the
        # longer propagation, without approaching the solver tolerance.
        direction_scale=0.5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
