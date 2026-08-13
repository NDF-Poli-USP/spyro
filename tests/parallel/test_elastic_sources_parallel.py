"""Elastic automated-adjoint tests under source parallelism.

Run with two MPI ranks::

    mpiexec -n 2 pytest tests/parallel/test_elastic_sources_parallel.py

With two sources and automatic parallelism, each ensemble member propagates
one source and records one vector-valued elastic shot. The ensemble reduced
functional then sums the three material gradients from both shots.
"""

import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


EXACT_MATERIAL = {"density": 0.1, "lambda": 0.025, "mu": 0.1}
# The ensemble reduction sums per-shot gradients and is agnostic to how the
# material is parameterized, so a single family is enough here. The difference
# between the Lame and velocity forms is covered serially, and much more
# cheaply, by tests/on_one_core/test_elastic_auto_adj_2d.py.
MATERIAL_CASES = [
    pytest.param(
        {"density": 0.12, "lambda": 0.20, "mu": 0.08},
        ("rho", "lmbda", "mu"),
        1.0,
        id="lame",
    ),
]


def make_dictionary(material_parameters):
    """Build a small two-source elastic problem for two MPI ranks."""
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
            "source_locations": [(-0.1, 0.3), (-0.1, 0.7)],
            "frequency": 10.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": spyro.create_transect(
                (-0.2, 0.2),
                (-0.2, 0.8),
                4,
            ),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.8,
            "dt": 0.002,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": "results/forward_output.pvd",
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": None,
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


@pytest.fixture(scope="module")
def exact_wave():
    """Compute one observed elastic shot on each ensemble member."""
    wave = spyro.IsotropicWave(make_dictionary(EXACT_MATERIAL))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.1, "periodic": True}
    )
    wave.forward_solve()
    return wave


def build_directions(wave, controls, relative_scale):
    """Build equal-relative smooth directions on every ensemble member."""
    common_shape = 1.0 + 0.1 * fire.sin(2.0 * wave.mesh_x) * fire.cos(
        2.0 * wave.mesh_z
    )
    return [
        fire.Function(control.function_space()).interpolate(
            relative_scale * control * common_shape
        )
        for control in controls.values()
    ]


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
def test_elastic_forward_sources_parallel(exact_wave):
    """Each ensemble member must retain its vector-valued local shot."""
    comm = exact_wave.comm

    assert comm.ensemble_comm.size == 2
    assert comm.comm.size == 1
    assert exact_wave.number_of_sources == 2
    assert exact_wave.current_sources == exact_wave.shot_ids_per_propagation[
        comm.ensemble_comm.rank
    ]

    number_of_steps = int(exact_wave.final_time / exact_wave.dt) + 1
    assert exact_wave.forward_solution_receivers.shape == (
        number_of_steps,
        4,
        exact_wave.dimension,
    )


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
@pytest.mark.parametrize(
    ("guess_material", "control_attributes", "relative_direction_scale"),
    MATERIAL_CASES,
)
def test_elastic_gradient_sources_parallel(
    exact_wave,
    guess_material,
    control_attributes,
    relative_direction_scale,
):
    """Taylor-test the three ensemble-summed elastic material gradients."""
    wave = spyro.IsotropicWave(make_dictionary(guess_material))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.1, "periodic": True}
    )
    wave.real_shot_record = exact_wave.forward_solution_receivers.copy()
    wave.enable_automated_adjoint()

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        assert wave.automated_adjoint.ensemble is wave.comm

        controls = wave.automated_adjoint.controls
        assert len(controls) == 3
        assert all(
            control is getattr(wave, attribute)
            for control, attribute in zip(controls.values(), control_attributes)
        )

        reduced_functional = (
            wave.automated_adjoint.create_reduced_functional(
                wave.functional_value
            )
        )
        assert isinstance(
            reduced_functional,
            fire_ad.EnsembleReducedFunctional,
        )

        gradients = wave.automated_adjoint.compute_gradient()
        assert tuple(gradients) == tuple(controls)
        assert all(
            isinstance(gradient, fire.Function)
            for gradient in gradients.values()
        )

        for gradient, control in zip(gradients.values(), controls.values()):
            assert gradient.function_space() == control.function_space()
            norms = wave.comm.ensemble_comm.allgather(fire.norm(gradient))
            assert np.allclose(norms, norms[0])

        convergence_rate = wave.automated_adjoint.verify_gradient(
            controls,
            direction=build_directions(
                wave,
                controls,
                relative_direction_scale,
            ),
        )
        assert convergence_rate > 1.9, (
            "Parallel elastic Taylor convergence rate %.4f < 1.90."
            % convergence_rate
        )
    finally:
        wave.automated_adjoint.clear_tape()

    assert wave.automated_adjoint._tape is None


if __name__ == "__main__":
    pytest.main([__file__])
