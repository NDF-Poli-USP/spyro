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
    control_parameters=None,
    relative_scale=1.0,
    minimum_rate=1.9,
):
    """Record a forward solve and verify all selected material gradients.

    The perturbation direction is random, as in the acoustic Taylor tests, but
    scaled by each control instead of being a single absolute field. Isotropic
    elastic controls differ in magnitude by more than an order of magnitude
    (``rho = 0.12`` against ``c = 1.73`` here), so one absolute direction would
    perturb density by a few hundred per cent while barely moving the wave
    speeds: the test would measure the density direction alone, and its large
    relative perturbation is what pushes the coarsest Taylor step out of the
    quadratic regime. Scaling by the control makes ``relative_scale`` a
    dimensionless perturbation that means the same thing for every parameter.
    """
    wave_guess = spyro.IsotropicWave(dictionary)
    wave_guess.set_mesh(input_mesh_parameters=mesh_parameters)
    wave_guess.real_shot_record = real_shot_record
    wave_guess.enable_automated_adjoint(controls=control_parameters)

    try:
        wave_guess.forward_solve()

        assert isinstance(wave_guess.automated_adjoint._tape, Tape), (
            "Pyadjoint tape is not a Tape instance after forward solve."
        )
        assert isinstance(wave_guess.functional_value, AdjFloat), (
            f"Expected wave_guess.functional_value to be an AdjFloat, "
            f"got {type(wave_guess.functional_value)}."
        )

        # The control selection is stored only on the automated adjoint, and
        # must still reference the very fields used by the variational form.
        controls = wave_guess.automated_adjoint.controls
        assert len(controls) == len(control_attributes), (
            f"Expected {len(control_attributes)} elastic controls, "
            f"got {len(controls)}."
        )
        assert all(
            control is getattr(wave_guess, attribute)
            for control, attribute in zip(controls.values(), control_attributes)
        ), "Automated-adjoint controls do not match the active parameterization."

        gradients = wave_guess.gradient_solve()
        assert tuple(gradients) == tuple(controls)
        assert all(
            isinstance(gradient, fire.Function)
            for gradient in gradients.values()
        )

        rng = np.random.default_rng(seed)
        direction = [
            fire.Function(
                control.function_space(),
                val=(
                    relative_scale
                    * control.dat.data_ro
                    * rng.random(control.dat.data_ro.shape)
                ),
            )
            for control in controls.values()
        ]
        convergence_rate = wave_guess.automated_adjoint.verify_gradient(
            controls,
            direction,
            dJdm=gradients,
        )
        # Narrowing the selection must not be checked here: ``taylor_test``
        # leaves the controls perturbed, so any gradient recomputed after it
        # is taken at a different point. It is covered without a solve by
        # tests/on_one_core/test_functional_computations.py.
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
    ("guess_material", "control_parameters", "control_attributes"),
    [
        pytest.param(
            {"density": 0.12, "lambda": 0.20, "mu": 0.08},
            None,
            ("rho", "lmbda", "mu"),
            id="lame",
        ),
        pytest.param(
            {
                "density": 0.12,
                "p_wave_velocity": np.sqrt(3.0),
                "s_wave_velocity": np.sqrt(2.0 / 3.0),
            },
            None,
            ("rho", "c", "c_s"),
            id="velocity",
        ),
    ],
)
def test_elastic_automated_adjoint_2d(
    exact_receiver_data,
    guess_material,
    control_parameters,
    control_attributes,
):
    """Taylor test for the automated adjoint of the 2D isotropic elastic wave.

    Both supported parameterizations are covered because they build genuinely
    different variational forms: the Lame one uses the moduli directly, while
    the velocity one differentiates through the UFL conversion
    ``lambda = rho*(c**2 - 2*c_s**2)`` recorded inside the form. Declaring the
    material in one family and inverting in the other does not add a third
    form, so it is covered by the much cheaper checks in ``test_fwi_controls``
    and by ``test_elastic_change_of_variables_matches_a_directly_taped_gradient``.

    Runs the following workflow:

    1. Solve the exact forward problem and store the
       synthetic receiver data as the "observed" record.
    2. Set up a guess model and enable the automated adjoint, so that pyadjoint
       records the computation on a tape.
    3. Run the guess forward solve; the L2 misfit functional is accumulated
       per time step as a pyadjoint-annotated AdjFloat.
    4. Build the reduced functional for the selected material controls.
    5. Verify the automated-adjoint gradient with a perturbation direction and
       check that the convergence rate exceeds 1.95 (second-order accuracy).
    """
    run_taylor_test(
        make_dictionary(guess_material),
        {"edge_length": 0.05, "periodic": True},
        exact_receiver_data,
        control_attributes,
        seed=42,
        control_parameters=control_parameters,
        minimum_rate=1.95,
    )


def compute_gradients(dictionary, mesh_parameters, real_shot_record, *,
                      controls, request):
    """Return the gradients a fresh solver produces for one control request."""
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters=mesh_parameters)
    wave.real_shot_record = real_shot_record
    wave.enable_automated_adjoint(controls=controls)
    try:
        wave.forward_solve()
        return {
            parameter: gradient.copy(deepcopy=True)
            for parameter, gradient in wave.gradient_solve(
                controls=request,
            ).items()
        }
    finally:
        wave.automated_adjoint.clear_tape()


@pytest.mark.slow
def test_elastic_change_of_variables_matches_a_directly_taped_gradient(
    exact_receiver_data,
):
    """Velocity gradients converted from Lame ones need no second solve.

    The equation is recorded once in each family for the same material, and
    the velocity gradients obtained by the chain rule from the Lame tape must
    reproduce those differentiated directly with respect to the velocities.
    """
    mesh_parameters = {"edge_length": 0.1, "periodic": True}
    velocity_parameters = [
        spyro.ElasticMaterialParameter.DENSITY,
        spyro.ElasticMaterialParameter.P_WAVE_VELOCITY,
        spyro.ElasticMaterialParameter.S_WAVE_VELOCITY,
    ]

    converted = compute_gradients(
        make_dictionary({"density": 0.12, "lambda": 0.20, "mu": 0.08}),
        mesh_parameters,
        exact_receiver_data,
        controls=None,
        request=velocity_parameters,
    )
    # The same material, declared through the velocities it implies.
    density, lmbda, mu = 0.12, 0.20, 0.08
    taped = compute_gradients(
        make_dictionary({
            "density": density,
            "p_wave_velocity": np.sqrt((lmbda + 2.0 * mu) / density),
            "s_wave_velocity": np.sqrt(mu / density),
        }),
        mesh_parameters,
        exact_receiver_data,
        controls=None,
        request=velocity_parameters,
    )

    assert tuple(converted) == tuple(taped)
    for parameter in taped:
        reference = taped[parameter].dat.data_ro
        assert np.allclose(
            converted[parameter].dat.data_ro,
            reference,
            rtol=1e-6,
            atol=1e-8 * np.abs(reference).max(),
        ), f"Converted gradient for {parameter.value} does not match."


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


# The record has to be long enough for the boundary to act on it, otherwise
# the ABC terms barely contribute to the functional and a wrong adjoint for
# them would still pass. With the source at (-0.1, 0.5), a unit domain and
# c_p = sqrt(3), c_s = sqrt(2/3), the round trip to the receivers at z = -0.2
# takes 0.36 s for the P wave reflected at the top, 0.56 s laterally, and
# 1.06 s for the *S* wave laterally. Elastic ABCs treat P and S with different
# operators, so the window must reach past that last arrival.
LONG_ENOUGH_FOR_S_WAVE = {"dt": 0.002, "final_time": 1.4}
BOUNDARY_CASES = [
    pytest.param(
        LAME_GUESS,
        {},
        ("rho", "lmbda", "mu"),
        LONG_ENOUGH_FOR_S_WAVE,
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
        LONG_ENOUGH_FOR_S_WAVE,
        id="homogeneous-dirichlet",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "backward"),
        ("rho", "c", "c_s"),
        LONG_ENOUGH_FOR_S_WAVE,
        id="stacey-backward",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("CE_A1", "backward"),
        ("rho", "c", "c_s"),
        LONG_ENOUGH_FOR_S_WAVE,
        id="clayton-engquist-a1-backward",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "central"),
        ("rho", "c", "c_s"),
        LONG_ENOUGH_FOR_S_WAVE,
        id="stacey-central",
    ),
    pytest.param(
        VELOCITY_GUESS,
        nrbc_settings("Stacey", "backward_2nd"),
        ("rho", "c", "c_s"),
        # This scheme is unstable at dt = 0.002 (the solver aborts), so it
        # needs twice as many steps as the others for the same window. The
        # record is trimmed to just past the lateral S arrival, which keeps
        # about 96% of the received energy at 86% of the cost.
        {"dt": 0.001, "final_time": 1.2},
        id="stacey-backward-2nd",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "guess_material",
        "boundary_settings",
        "control_attributes",
        "time_axis",
    ),
    BOUNDARY_CASES,
)
def test_elastic_automated_adjoint_boundary_conditions(
    guess_material,
    boundary_settings,
    control_attributes,
    time_axis,
):
    """Taylor-test supported elastic boundary formulations and time schemes.

    Each case pairs an absorbing or essential boundary formulation with the
    time discretization it needs, and runs long enough for that boundary to
    act on the record; see :data:`LONG_ENOUGH_FOR_S_WAVE`.
    """
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
    dictionary["time_axis"].update(time_axis)

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
    )


if __name__ == "__main__":
    pytest.main([__file__])
