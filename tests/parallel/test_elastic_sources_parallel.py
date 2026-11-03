"""Elastic automated-adjoint test under source parallelism.

Run with two MPI ranks::

    mpiexec -n 2 pytest tests/parallel/test_elastic_sources_parallel.py
"""

import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


Parameter = spyro.ElasticMaterialParameter
EXACT_MATERIAL = {"density": 0.1, "lambda": 0.025, "mu": 0.1}
GUESS_MATERIAL = {"density": 0.12, "lambda": 0.20, "mu": 0.08}


def make_dictionary(material_parameters):
    """Build a two-source elastic problem for two MPI ranks.

    Parameters
    ----------
    material_parameters : dict
        Complete Lame material parameterization.

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
            "source_locations": [(-0.1, 0.3), (-0.1, 0.7)],
            "frequency": 10.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": spyro.create_transect(
                (-0.2, 0.2), (-0.2, 0.8), 4,
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
def exact_wave():
    """Compute one observed elastic shot on each ensemble member."""
    wave = spyro.IsotropicWave(make_dictionary(EXACT_MATERIAL))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.1, "periodic": True},
    )
    wave.forward_solve()
    return wave


@pytest.mark.newer_firedrake
@pytest.mark.slow
@pytest.mark.parallel(2)
def test_elastic_gradient_sources_parallel(exact_wave):
    """Taylor-test the ensemble-summed multi-control elastic gradient."""
    wave = spyro.IsotropicWave(make_dictionary(GUESS_MATERIAL))
    wave.set_mesh(
        input_mesh_parameters={"edge_length": 0.1, "periodic": True},
    )
    wave.real_shot_record = exact_wave.forward_solution_receivers.copy()
    wave.enable_automated_adjoint()

    try:
        wave.forward_solve()

        assert wave.comm.ensemble_comm.size == 2
        assert wave.comm.comm.size == 1
        assert wave.current_sources == wave.shot_ids_per_propagation[
            wave.comm.ensemble_comm.rank
        ]
        number_of_steps = int(wave.final_time / wave.dt) + 1
        assert wave.forward_solution_receivers.shape == (
            number_of_steps, 4, wave.dimension,
        )
        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)

        reduced_functional = wave.automated_adjoint.create_reduced_functional(
            wave.functional_value,
        )
        assert isinstance(
            reduced_functional, fire_ad.EnsembleReducedFunctional,
        )

        gradients = wave.gradient_solve()
        assert tuple(gradients) == (
            Parameter.DENSITY,
            Parameter.LAMBDA,
            Parameter.MU,
        )
        for gradient in gradients.values():
            norms = wave.comm.ensemble_comm.allgather(fire.norm(gradient))
            assert np.allclose(norms, norms[0])

        common_shape = 1.0 + 0.1 * fire.sin(2.0 * wave.mesh_x) * fire.cos(
            2.0 * wave.mesh_z,
        )
        directions = [
            fire.Function(control.function_space()).interpolate(
                control * common_shape,
            )
            for control in wave.automated_adjoint.controls
        ]
        convergence_rate = wave.automated_adjoint.verify_gradient(
            wave.automated_adjoint.controls,
            direction=directions,
            dJdm=gradients,
        )
        assert convergence_rate > 1.9, (
            "Parallel elastic Taylor convergence rate %.4f < 1.90."
            % convergence_rate
        )
    finally:
        wave.automated_adjoint.clear_tape()


if __name__ == "__main__":
    pytest.main([__file__])
