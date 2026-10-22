"""Taylor tests for the automated adjoint of the coupled acoustic-elastic wave.

The fluid and the solid are solved on two submeshes of one parent mesh, and
the elastic material parameters are fields on the solid submesh. These tests
check that the gradient with respect to those fields is consistent with the
functional recorded on the tape, for both the sequential and the monolithic
coupled solvers.
"""

from importlib.metadata import version

import firedrake as fire
import numpy as np
from packaging.version import Version
from pyadjoint import AdjFloat, Tape
import pytest

import spyro
from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave


pytestmark = [
    pytest.mark.newer_firedrake,
    pytest.mark.slow,
    # Annotating forms that integrate over more than one mesh, as the
    # interface terms between the submeshes do, needs Firedrake 2026.10.
    pytest.mark.skipif(
        Version(version("firedrake")) < Version("2026.10"),
        reason="Firedrake < 2026.10 cannot annotate multi-mesh forms.",
    ),
]

Parameter = spyro.ElasticMaterialParameter
TRUE_SOLID = {"density": 2.0, "p_wave_velocity": 2.0, "s_wave_velocity": 1.2}
GUESS_SOLID = {"density": 2.2, "p_wave_velocity": 2.2, "s_wave_velocity": 1.1}


def make_dictionary(solid: dict) -> dict:
    """Build a compact two-dimensional fluid-over-solid model.

    The fluid occupies the upper half of a unit square and the solid the
    lower half. The source and the receivers are in the fluid, just above
    the interface, so the recorded pressure carries the reflection from the
    solid.

    Parameters
    ----------
    solid : dict
        Solid ``density``, ``p_wave_velocity`` and ``s_wave_velocity``.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    return {
        "options": {
            "cell_type": "Q",
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
            "edge_length": 0.1,
            "interface_z": -0.5,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.4, 0.5)],
            "frequency": 5.0,
            "delay": 0.2,
            "delay_type": "time",
            "amplitude": 1.0,
            "receiver_locations": spyro.create_transect(
                (-0.45, 0.2), (-0.45, 0.8), 5,
            ),
            # The observed record is sampled the way the automated adjoint
            # samples the guess, so the two are aligned step by step.
            "use_vertex_only_mesh": True,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.3,
            "dt": 0.001,
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
            "velocity_fluid": None,
            "bulk_modulus": 2.25,
            "density_fluid": 1.0,
            "density_solid": solid["density"],
            "p_wave_velocity": solid["p_wave_velocity"],
            "s_wave_velocity": solid["s_wave_velocity"],
            "real_velocity_file": None,
        },
    }


@pytest.fixture(scope="module")
def exact_receiver_data() -> np.ndarray:
    """Compute one observed pressure record shared by every case.

    Returns
    -------
    numpy.ndarray
        Pressure at the fluid receivers, one row per time step.
    """
    wave = AcousticElasticWave(make_dictionary(TRUE_SOLID))
    wave.forward_solve()
    return wave.forward_solution_receivers


def test_elastic_parameters_are_solid_submesh_fields():
    """The elastic parameters are fields on the solid submesh only."""
    wave = AcousticElasticWave(make_dictionary(TRUE_SOLID))
    parameters = wave.initialize_physical_parameters()

    independent = parameters.select()
    assert tuple(independent) == (
        Parameter.DENSITY,
        Parameter.P_WAVE_VELOCITY,
        Parameter.S_WAVE_VELOCITY,
    )
    for name, field in independent.items():
        assert field.function_space().mesh() is wave.submesh_solid
        assert np.allclose(field.dat.data_ro, TRUE_SOLID[name.value])

    # The Lame parameters the elastic form reads are computed from them.
    mu = TRUE_SOLID["density"] * TRUE_SOLID["s_wave_velocity"]**2
    lmbda = TRUE_SOLID["density"] * TRUE_SOLID["p_wave_velocity"]**2 - 2 * mu
    dx_solid = wave.dx_solid
    area = fire.assemble(fire.Constant(1.0) * dx_solid)
    assert np.isclose(
        fire.assemble(parameters[Parameter.MU] * dx_solid) / area, mu,
    )
    assert np.isclose(
        fire.assemble(parameters[Parameter.LAMBDA] * dx_solid) / area, lmbda,
    )


@pytest.mark.parametrize("use_monolithic", [False, True],
                         ids=["sequential", "monolithic"])
def test_acoustic_elastic_automated_adjoint(
    exact_receiver_data: np.ndarray, use_monolithic: bool,
):
    """Taylor-test the gradient with respect to the solid parameters.

    Parameters
    ----------
    exact_receiver_data : numpy.ndarray
        Observed pressure record from the true model.
    use_monolithic : bool
        Whether to solve the coupled step monolithically or sequentially.
    """
    wave = AcousticElasticWave(make_dictionary(GUESS_SOLID))
    wave.use_monolithic = use_monolithic
    wave.real_shot_record = exact_receiver_data
    wave.enable_automated_adjoint()

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        expected_parameters = (
            Parameter.DENSITY,
            Parameter.P_WAVE_VELOCITY,
            Parameter.S_WAVE_VELOCITY,
        )
        assert tuple(wave.automated_adjoint.control_parameter_names) == (
            expected_parameters
        )
        for control in wave.automated_adjoint.controls:
            assert control.function_space().mesh() is wave.submesh_solid

        gradients = wave.gradient_solve()
        assert tuple(gradients) == expected_parameters
        for gradient in gradients.values():
            assert isinstance(gradient, fire.Function)
            assert gradient.function_space().mesh() is wave.submesh_solid
            assert fire.norm(gradient) > 0.0

        # Replaying the tape at the recorded controls has to give back the
        # recorded functional. Otherwise the tape has lost part of the
        # forward problem (the source, for instance), and the Taylor test
        # below would check the gradient of some other functional.
        replayed = wave.automated_adjoint.recompute_functional(
            wave.automated_adjoint.controls,
        )
        assert np.isclose(
            float(replayed), float(wave.functional_value), rtol=1e-10,
        )

        rng = np.random.default_rng(42)
        directions = [
            fire.Function(
                control.function_space(),
                val=0.05 * control.dat.data_ro * rng.random(
                    control.dat.data_ro.shape,
                ),
            )
            for control in wave.automated_adjoint.controls
        ]
        convergence_rate = wave.automated_adjoint.verify_gradient(
            wave.automated_adjoint.controls,
            direction=directions,
            dJdm=gradients,
        )
        assert convergence_rate > 1.9, (
            "Acoustic-elastic Taylor convergence rate %.4f < 1.90."
            % convergence_rate
        )
    finally:
        wave.automated_adjoint.clear_tape()


if __name__ == "__main__":
    pytest.main([__file__])
