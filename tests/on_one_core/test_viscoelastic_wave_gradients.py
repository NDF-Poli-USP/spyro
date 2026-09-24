"""Taylor tests for the viscoelastic wave propagators."""

import firedrake as fire
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


pytestmark = [pytest.mark.newer_firedrake, pytest.mark.slow]

ElasticParameter = spyro.ElasticMaterialParameter
AnisotropicParameter = spyro.AnisotropicMaterialParameter

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
    pytest.param(
        spyro.IsotropicWave,
        {},
        (
            ElasticParameter.DENSITY,
            ElasticParameter.P_WAVE_VELOCITY,
            ElasticParameter.S_WAVE_VELOCITY,
        ),
        id="isotropic",
    ),
    pytest.param(
        spyro.AnisotropicVTIWave,
        ANISOTROPY,
        (
            ElasticParameter.DENSITY,
            ElasticParameter.P_WAVE_VELOCITY,
            ElasticParameter.S_WAVE_VELOCITY,
            AnisotropicParameter.DELTA,
            AnisotropicParameter.EPSILON,
            AnisotropicParameter.GAMMA,
        ),
        id="vti",
    ),
    pytest.param(
        spyro.AnisotropicTTIWave,
        {**ANISOTROPY, "theta": 20.0, "phi": 0.0},
        (
            ElasticParameter.DENSITY,
            ElasticParameter.P_WAVE_VELOCITY,
            ElasticParameter.S_WAVE_VELOCITY,
            AnisotropicParameter.DELTA,
            AnisotropicParameter.EPSILON,
            AnisotropicParameter.GAMMA,
            AnisotropicParameter.THETA,
        ),
        id="tti",
    ),
]


def make_dictionary(
    material: dict,
    medium_parameters: dict,
    dimension: int = 2,
) -> dict:
    """Build a compact viscoelastic model.

    Parameters
    ----------
    material : dict
        Density and wave speeds for the model.
    medium_parameters : dict
        Anisotropy and orientation parameters required by the wave class.
    dimension : int, optional
        Spatial dimension. The default is two. Three dimensions select an
        extruded hexahedral mesh with the spectral method.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    is_3d = dimension == 3
    options = {
        "cell_type": "Q" if is_3d else "T",
        "variant": "lumped",
        "degree": 2,
        "dimension": dimension,
    }
    if is_3d:
        options["method"] = "spectral_quadrilateral"

    return {
        "options": options,
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 1.0 if is_3d else 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [
                (-0.2, 0.5, 0.5) if is_3d else (-0.2, 0.5)
            ],
            "frequency": 5.0,
            "delay": 0.1,
            "delay_type": "time",
            "amplitude": (
                np.array([0.0, 1.0, 0.0])
                if is_3d else np.array([0.0, 1.0])
            ),
            "receiver_locations": (
                [(-0.4, 0.5, 0.5)]
                if is_3d else [(-0.7, 0.25), (-0.7, 0.75)]
            ),
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
    dimension: int = 2,
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
    dimension : int, optional
        Spatial dimension. The default is two.

    Returns
    -------
    spyro.IsotropicWave
        Configured wave object whose forward problem has not yet been solved.
    """
    wave = wave_class(
        make_dictionary(material, medium_parameters, dimension=dimension),
    )
    wave.set_mesh(
        input_mesh_parameters={
            "edge_length": 0.5 if dimension == 3 else 0.2,
            "periodic": dimension == 2,
        },
    )
    return wave


@pytest.mark.parametrize(
    "wave_class, medium_parameters, control_parameters",
    WAVE_CASES,
)
def test_viscoelastic_gradient(
    wave_class: type[spyro.IsotropicWave],
    medium_parameters: dict,
    control_parameters: tuple,
) -> None:
    """Taylor-test selected physical gradients of a viscoelastic wave.

    Parameters
    ----------
    wave_class : type of spyro.IsotropicWave
        Wave class under test.
    medium_parameters : dict
        Anisotropy and orientation parameters required by the wave class.
    control_parameters : tuple of enum.Enum
        Continuous physical parameters included in the Taylor test.

    Returns
    -------
    None
        The test succeeds when the joint Taylor rate is greater than 1.9.
    """
    exact = build_wave(wave_class, EXACT_MATERIAL, medium_parameters)
    exact.forward_solve()

    wave = build_wave(wave_class, GUESS_MATERIAL, medium_parameters)
    wave.real_shot_record = exact.forward_solution_receivers
    wave.enable_automated_adjoint(control_parameters=control_parameters)

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        assert tuple(
            wave.automated_adjoint.control_parameter_names
        ) == control_parameters

        gradients = wave.gradient_solve()
        assert tuple(gradients) == control_parameters
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
        directional_derivatives = {
            parameter.value: fire.assemble(
                fire.inner(gradient, direction) * fire.dx,
            )
            for parameter, gradient, direction in zip(
                control_parameters,
                gradients.values(),
                directions,
            )
        }
        functional_scale = max(1.0, abs(float(wave.functional_value)))
        resolution = 100.0 * np.finfo(float).eps * functional_scale
        unresolved = {
            parameter: derivative
            for parameter, derivative in directional_derivatives.items()
            if abs(derivative) <= resolution
        }
        assert not unresolved, (
            "directional derivatives are below numerical resolution "
            f"({resolution:.3e}): {unresolved}"
        )

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


def test_tti_3d_phi_gradient() -> None:
    """Taylor-test the TTI azimuth gradient on a spectral extruded mesh.

    Returns
    -------
    None
        The test succeeds when the azimuth Taylor rate is greater than 1.9.
    """
    medium_parameters = {
        **ANISOTROPY,
        "theta": 20.0,
        "phi": 15.0,
    }
    exact = build_wave(
        spyro.AnisotropicTTIWave,
        EXACT_MATERIAL,
        medium_parameters,
        dimension=3,
    )
    exact.forward_solve()

    wave = build_wave(
        spyro.AnisotropicTTIWave,
        GUESS_MATERIAL,
        medium_parameters,
        dimension=3,
    )
    assert wave.mesh.extruded
    assert wave.method == "spectral_quadrilateral"
    wave.real_shot_record = exact.forward_solution_receivers
    controls = (AnisotropicParameter.PHI,)
    wave.enable_automated_adjoint(control_parameters=controls)

    try:
        wave.forward_solve()
        gradients = wave.gradient_solve()
        assert tuple(gradients) == controls
        direction = fire.Function(
            wave.phi.function_space(),
            val=0.1 * wave.phi.dat.data_ro,
        )
        directional_derivative = fire.assemble(
            fire.inner(
                gradients[AnisotropicParameter.PHI],
                direction,
            ) * fire.dx,
        )
        functional_scale = max(1.0, abs(float(wave.functional_value)))
        resolution = 100.0 * np.finfo(float).eps * functional_scale
        assert abs(directional_derivative) > resolution, (
            "phi directional derivative is below numerical resolution "
            f"({resolution:.3e}): {directional_derivative:.3e}"
        )

        convergence_rate = wave.automated_adjoint.verify_gradient(
            wave.automated_adjoint.controls,
            direction=direction,
            dJdm=gradients,
        )
        assert convergence_rate > 1.9, (
            "TTI azimuth Taylor convergence rate "
            f"{convergence_rate:.4f} < 1.90."
        )
    finally:
        wave.automated_adjoint.clear_tape()
