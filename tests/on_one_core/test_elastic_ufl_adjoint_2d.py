"""Taylor tests for the UFL-derived implemented adjoint of the elastic wave."""

from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType


Parameter = spyro.ElasticMaterialParameter
LAME_MATERIAL = {"density": 1.0, "lambda": 4.0, "mu": 1.0}
VELOCITY_MATERIAL = {
    "density": 1.0,
    "p_wave_velocity": np.sqrt(6.0),
    "s_wave_velocity": 1.0,
}


def make_dictionary(material_parameters, use_vertex_only_mesh=True):
    """Build a compact two-dimensional elastic model.

    Parameters
    ----------
    material_parameters : dict
        Complete Lame or velocity material parameterization.
    use_vertex_only_mesh : bool, optional
        Whether receivers (and sources) use the vertex-only mesh rather than
        the Dirac delta projection.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
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
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
            "use_vertex_only_mesh": use_vertex_only_mesh,
        },
        "synthetic_data": {
            "type": "object",
            **material_parameters,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.5,
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
    }


def solve_forward(dictionary, real_shot_record=None, adjoint=False):
    """Run one elastic forward solve on a coarse mesh.

    Parameters
    ----------
    dictionary : dict
        Spyro model dictionary.
    real_shot_record : numpy.ndarray, optional
        Observed receiver data to set on the solver.
    adjoint : bool, optional
        Whether to enable the UFL-derived adjoint before solving, so the
        forward solution is stored.

    Returns
    -------
    spyro.IsotropicWave
        The solved wave object.
    """
    wave = spyro.IsotropicWave(dictionary=deepcopy(dictionary))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    if adjoint:
        wave.enable_implemented_adjoint(
            adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        )
    wave.forward_solve()
    return wave


def functional(wave, real_shot_record):
    """Return the L2 misfit functional of the last forward solve.

    Parameters
    ----------
    wave : spyro.IsotropicWave
        Solved wave object.
    real_shot_record : numpy.ndarray
        Observed receiver data.

    Returns
    -------
    float
        Functional value.
    """
    return spyro.utils.compute_functional(
        wave, real_shot_record - wave.forward_solution_receivers,
    )


def taylor_convergence_rates(wave, real_shot_record, gradient, parameter):
    """Perturb one parameter and return the Taylor remainder rates.

    Parameters
    ----------
    wave : spyro.IsotropicWave
        Solved guess wave whose ``gradient`` was just computed.
    real_shot_record : numpy.ndarray
        Observed receiver data.
    gradient : firedrake.Function
        Gradient with respect to ``parameter``.
    parameter : ElasticMaterialParameter
        Parameter to perturb.

    Returns
    -------
    tuple
        ``(rates, directional_errors)``: the convergence rates of the
        first-order Taylor remainder between consecutive steps, and the
        relative error of the finite-difference directional derivative.
    """
    base_functional = functional(wave, real_shot_record)
    field = wave.physical_parameters[parameter]
    control_space = field.function_space()
    values = np.random.default_rng(5).random(control_space.dim()) - 0.5
    direction = fire.Function(control_space, val=values)
    direction.assign(direction / fire.norm(direction))
    directional_derivative = fire.assemble(
        gradient * direction * fire.dx(**wave.quadrature_rule)
    )
    base = fire.Function(control_space).assign(field)

    steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    remainders = []
    directional_errors = []
    for step in steps:
        wave.physical_parameters.update(parameter, base + step * direction)
        wave.forward_solve()
        perturbed_functional = functional(wave, real_shot_record)
        finite_difference = (perturbed_functional - base_functional) / step
        remainders.append(abs(
            perturbed_functional
            - base_functional
            - step * directional_derivative
        ))
        directional_errors.append(abs(
            (finite_difference - directional_derivative)
            / directional_derivative
        ))
    remainders = np.array(remainders)
    rates = np.log(remainders[:-1] / remainders[1:]) / np.log(
        steps[:-1] / steps[1:]
    )
    return rates, np.array(directional_errors)


def test_elastic_ufl_derived_adjoint_uses_forward_residual_form():
    exact_model = make_dictionary(LAME_MATERIAL | {"density": 1.2})
    real_shot_record = solve_forward(exact_model).forward_solution_receivers

    guess = solve_forward(
        make_dictionary(LAME_MATERIAL), real_shot_record, adjoint=True,
    )
    gradients = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    assert guess.forward_residual_form is not None
    assert guess.forward_residual_states is not None
    assert guess.get_adjoint_source().function_space() == (
        guess.source_function.function_space()
    )
    assert set(gradients) == {
        Parameter.DENSITY, Parameter.LAMBDA, Parameter.MU,
    }
    assert all(
        isinstance(gradient, fire.Function) for gradient in gradients.values()
    )
    assert all(
        np.isfinite(fire.norm(gradient)) for gradient in gradients.values()
    )


@pytest.mark.parametrize("use_vertex_only_mesh", [True, False])
@pytest.mark.parametrize(
    "parameter", [Parameter.DENSITY, Parameter.LAMBDA, Parameter.MU],
)
def test_elastic_ufl_derived_adjoint_taylor_remainder(
    parameter, use_vertex_only_mesh,
):
    """Second-order Taylor remainder for each Lame-parameterization control,
    with the misfit injected through either receiver interpolation."""
    exact_material = dict(LAME_MATERIAL)
    exact_material[parameter.value] *= 1.2
    exact_model = make_dictionary(exact_material, use_vertex_only_mesh)
    real_shot_record = solve_forward(exact_model).forward_solution_receivers

    guess = solve_forward(
        make_dictionary(LAME_MATERIAL, use_vertex_only_mesh),
        real_shot_record,
        adjoint=True,
    )
    gradients = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    rates, directional_errors = taylor_convergence_rates(
        guess, real_shot_record, gradients[parameter], parameter,
    )
    assert np.all(rates > 1.8), rates
    # The finite-difference error halves with the step; the smallest step
    # pins the directional derivative itself.
    assert directional_errors[-1] < 0.1, directional_errors


def test_elastic_ufl_derived_adjoint_selects_controls():
    """A subset of the physical parameters, in the velocity parameterization
    the equation is rewritten in, is a valid selection."""
    exact_model = make_dictionary(VELOCITY_MATERIAL | {"s_wave_velocity": 1.2})
    real_shot_record = solve_forward(exact_model).forward_solution_receivers

    guess = solve_forward(
        make_dictionary(VELOCITY_MATERIAL), real_shot_record, adjoint=True,
    )
    gradients = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        control_parameters={Parameter.S_WAVE_VELOCITY},
    )
    assert set(gradients) == {Parameter.S_WAVE_VELOCITY}

    rates, directional_errors = taylor_convergence_rates(
        guess, real_shot_record, gradients[Parameter.S_WAVE_VELOCITY],
        Parameter.S_WAVE_VELOCITY,
    )
    assert np.all(rates > 1.8), rates
    assert directional_errors[-1] < 0.1, directional_errors


def test_elastic_ufl_derived_adjoint_rejects_computed_controls():
    guess = solve_forward(make_dictionary(LAME_MATERIAL), adjoint=True)
    guess.real_shot_record = guess.forward_solution_receivers

    with pytest.raises(TypeError, match="computed from the other physical"):
        guess.gradient_solve(
            adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
            control_parameters={Parameter.P_WAVE_VELOCITY},
        )


def test_elastic_hand_derived_adjoint_is_not_implemented():
    guess = solve_forward(make_dictionary(LAME_MATERIAL))

    with pytest.raises(NotImplementedError, match="UFL-derived"):
        guess.gradient_solve(adjoint_type=AdjointType.IMPLEMENTED_ADJOINT)


if __name__ == "__main__":
    pytest.main([__file__])
