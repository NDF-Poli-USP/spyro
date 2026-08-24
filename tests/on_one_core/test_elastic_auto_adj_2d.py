"""Taylor tests for selectable isotropic-elastic material controls."""

import firedrake as fire
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


pytestmark = [pytest.mark.newer_firedrake, pytest.mark.slow]

Parameter = spyro.ElasticMaterialParameter
LAME_MATERIAL = {"density": 0.12, "lambda": 0.20, "mu": 0.08}
VELOCITY_MATERIAL = {
    "density": 0.12,
    "p_wave_velocity": np.sqrt(3.0),
    "s_wave_velocity": np.sqrt(2.0 / 3.0),
}


def make_dictionary(material_parameters):
    """Build a compact two-dimensional elastic model.

    Parameters
    ----------
    material_parameters : dict
        Complete Lame or velocity material parameterization.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 4,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
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
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.2), (-0.8, 0.8), 10,
            ),
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
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            **material_parameters,
            "real_velocity_file": None,
        },
    }


@pytest.fixture(scope="module")
def exact_receiver_data():
    """Compute one observed record shared by all parameterizations."""
    wave = spyro.IsotropicWave(
        make_dictionary({"density": 0.1, "lambda": 0.025, "mu": 0.1}),
    )
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.1, "periodic": True},
    )
    wave.forward_solve()
    return wave.forward_solution_receivers


LAME = spyro.ElasticMaterialParameterization.LAME
VELOCITY = spyro.ElasticMaterialParameterization.VELOCITY

# guess material, parameterization to write the equation in, controls to
# select within it, controls expected on the adjoint.
CASES = [
    pytest.param(
        LAME_MATERIAL,
        None,
        None,
        (Parameter.DENSITY, Parameter.LAMBDA, Parameter.MU),
        id="lame-default",
    ),
    pytest.param(
        VELOCITY_MATERIAL,
        None,
        None,
        (
            Parameter.DENSITY,
            Parameter.P_WAVE_VELOCITY,
            Parameter.S_WAVE_VELOCITY,
        ),
        id="velocity-default",
    ),
    pytest.param(
        LAME_MATERIAL,
        VELOCITY,
        {Parameter.S_WAVE_VELOCITY},
        (Parameter.S_WAVE_VELOCITY,),
        id="lame-model-rewritten-in-velocity",
    ),
    pytest.param(
        VELOCITY_MATERIAL,
        LAME,
        {Parameter.LAMBDA, Parameter.MU},
        (Parameter.LAMBDA, Parameter.MU),
        id="velocity-model-rewritten-in-lame",
    ),
]


@pytest.mark.parametrize(
    (
        "guess_material",
        "parameterization",
        "control_parameters",
        "expected_parameters",
    ),
    CASES,
)
def test_elastic_automated_adjoint_controls(
    exact_receiver_data,
    guess_material,
    parameterization,
    control_parameters,
    expected_parameters,
):
    """Taylor-test full and subset controls, in either parameterization."""
    wave = spyro.IsotropicWave(make_dictionary(guess_material))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.05, "periodic": True},
    )
    wave.real_shot_record = exact_receiver_data
    if parameterization is not None:
        # Which family the equation is written in is decided on the wave,
        # before anything selects controls within it.
        wave.initialize_physical_parameters()
        wave.set_physical_parameterization(parameterization)
    wave.enable_automated_adjoint()
    if control_parameters is not None:
        wave.automated_adjoint.set_control_parameters(control_parameters)

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        assert tuple(wave.automated_adjoint.control_parameter_names) == (
            expected_parameters
        )

        gradients = wave.gradient_solve()
        assert tuple(gradients) == expected_parameters
        assert all(
            isinstance(gradient, fire.Function)
            for gradient in gradients.values()
        )

        rng = np.random.default_rng(42)
        directions = [
            fire.Function(
                control.function_space(),
                val=control.dat.data_ro * rng.random(control.dat.data_ro.shape),
            )
            for control in wave.automated_adjoint.controls
        ]
        convergence_rate = wave.automated_adjoint.verify_gradient(
            wave.automated_adjoint.controls,
            direction=directions,
            dJdm=gradients,
        )
        assert convergence_rate > 1.9, (
            "Elastic Taylor convergence rate %.4f < 1.90."
            % convergence_rate
        )
    finally:
        wave.automated_adjoint.clear_tape()


if __name__ == "__main__":
    pytest.main([__file__])
