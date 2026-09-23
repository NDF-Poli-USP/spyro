"""Taylor tests for the viscoelastic wave propagators."""

import firedrake as fire
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


pytestmark = [pytest.mark.newer_firedrake, pytest.mark.slow]

Parameter = spyro.ElasticMaterialParameter
CONTROLS = (Parameter.P_WAVE_VELOCITY, Parameter.S_WAVE_VELOCITY)

EXACT_MATERIAL = {
    "density": 1.0,
    "p_wave_velocity": 1.5,
    "s_wave_velocity": 0.8,
}
GUESS_MATERIAL = {
    "density": 1.0,
    "p_wave_velocity": 1.7,
    "s_wave_velocity": 0.9,
}
ANISOTROPY = {
    "epsilon": 0.10,
    "gamma": 0.05,
    "delta": 0.08,
    "anisotropy": "exact",
}
ATTENUATION = {
    "Q_vp": 0.05,
    "Q_vs": 0.05,
    "Q_epsilon": 0.05,
    "Q_delta": 0.05,
    "Q_gamma": 0.05,
}

WAVE_CASES = [
    pytest.param(spyro.IsotropicWave, {}, id="isotropic"),
    pytest.param(spyro.AnisotropicVTIWave, ANISOTROPY, id="vti"),
    pytest.param(
        spyro.AnisotropicTTIWave,
        {**ANISOTROPY, "theta": 20.0, "phi": 0.0},
        id="tti",
    ),
]


def make_dictionary(
    material: dict,
    medium_parameters: dict,
) -> dict:
    """Build a compact viscoelastic model.

    Parameters
    ----------
    material : dict
        Density and wave speeds for the model.
    medium_parameters : dict
        Anisotropy and orientation parameters required by the wave class.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
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
            "source_locations": [(-0.2, 0.5)],
            "frequency": 5.0,
            "delay": 0.1,
            "delay_type": "time",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.7, 0.25), (-0.7, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.3,
            "dt": 0.002,
            "output_frequency": 1000,
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
            **material,
            **medium_parameters,
            **ATTENUATION,
            "real_velocity_file": None,
        },
        "viscoelastic": True,
        "viscoelasticity": {
            "visco_type": "maxwell_gsls_Q",
            "Q_type": "constant",
            "branches": 1,
            "omega_gsls": [20.0],
            "y_gsls": [0.1],
        },
    }


def build_wave(
    wave_class: type[spyro.IsotropicWave],
    material: dict,
    medium_parameters: dict,
) -> spyro.IsotropicWave:
    """Create a viscoelastic wave on a small periodic mesh.

    Parameters
    ----------
    wave_class : type of spyro.IsotropicWave
        Wave class under test.
    material : dict
        Density and wave speeds for the model.
    medium_parameters : dict
        Anisotropy and orientation parameters required by the wave class.

    Returns
    -------
    spyro.IsotropicWave
        Configured wave object whose forward problem has not yet been solved.
    """
    wave = wave_class(make_dictionary(material, medium_parameters))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.2, "periodic": True},
    )
    return wave


@pytest.mark.parametrize("wave_class, medium_parameters", WAVE_CASES)
def test_viscoelastic_gradient(
    wave_class: type[spyro.IsotropicWave],
    medium_parameters: dict,
) -> None:
    """Taylor-test the wave-speed gradients of a viscoelastic wave.

    Parameters
    ----------
    wave_class : type of spyro.IsotropicWave
        Wave class under test.
    medium_parameters : dict
        Anisotropy and orientation parameters required by the wave class.
    """
    exact = build_wave(wave_class, EXACT_MATERIAL, medium_parameters)
    exact.forward_solve()

    wave = build_wave(wave_class, GUESS_MATERIAL, medium_parameters)
    wave.real_shot_record = exact.forward_solution_receivers
    wave.enable_automated_adjoint(control_parameters=CONTROLS)

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        assert tuple(wave.automated_adjoint.control_parameter_names) == CONTROLS

        gradients = wave.gradient_solve()
        assert tuple(gradients) == CONTROLS
        assert all(
            isinstance(gradient, fire.Function)
            for gradient in gradients.values()
        )

        rng = np.random.default_rng(42)
        directions = [
            fire.Function(
                control.function_space(),
                val=0.1
                * control.dat.data_ro
                * rng.random(control.dat.data_ro.shape),
            )
            for control in wave.automated_adjoint.controls
        ]
        convergence_rate = wave.automated_adjoint.verify_gradient(
            wave.automated_adjoint.controls,
            direction=directions,
            dJdm=gradients,
        )
        assert convergence_rate > 1.9, (
            "Viscoelastic Taylor convergence rate "
            f"{convergence_rate:.4f} < 1.90."
        )
    finally:
        wave.automated_adjoint.clear_tape()
