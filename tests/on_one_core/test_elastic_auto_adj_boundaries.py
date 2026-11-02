"""Taylor tests for the elastic automated adjoint under each boundary condition.

The other elastic automated-adjoint tests run on a periodic mesh, which has
no boundary at all. The isotropic solver also offers a traction-free
boundary, Dirichlet conditions read from ``boundary_conditions`` and the
local absorbing conditions of :mod:`spyro.solvers.elastic_wave.local_abc`
(Stacey or Clayton-Engquist A1, with a backward, second-order backward or
central time derivative). The absorbing terms are written in the wave
speeds and the density, so they depend on the controls and on up to three
past displacement fields; the Dirichlet conditions enter the solve block.
Each has to be recorded and differentiated for the gradient to be right,
which is what these tests check.
"""

import firedrake as fire
import numpy as np
from pyadjoint import AdjFloat, Tape
import pytest

import spyro


pytestmark = [pytest.mark.newer_firedrake, pytest.mark.slow]

Parameter = spyro.ElasticMaterialParameter
EXACT_MATERIAL = {"density": 0.1, "lambda": 0.025, "mu": 0.1}
LAME_MATERIAL = {"density": 0.12, "lambda": 0.20, "mu": 0.08}
# The same medium as LAME_MATERIAL, declared through the wave speeds.
VELOCITY_MATERIAL = {
    "density": 0.12,
    "p_wave_velocity": np.sqrt(3.0),
    "s_wave_velocity": np.sqrt(2.0 / 3.0),
}

# Deliberately small: the gradient check is exact, so it does not need a
# well-resolved model - only enough time for the P wave to reach every side
# of the domain and come back to the receivers.
FINAL_TIME = 0.8
EDGE_LENGTH = 0.1
DT = 0.002

# Boundary ids of spyro's rectangle: 1 top (z = 0), 2 bottom, 3 left, 4 right.
SIDES = (1, 2, 3, 4)

# spyro's default solve is one Jacobi sweep, exact for the lumped mass
# matrix alone. This actually solves the linear system, for the case whose
# operator is not diagonal.
EXACT_SOLVE = {
    "ksp_type": "cg",
    "pc_type": "jacobi",
    "ksp_rtol": 1e-12,
    "ksp_atol": 1e-14,
}


def local_abc(kind: str, dt_scheme: str) -> dict:
    """Build the ``absorving_boundary_conditions`` entry of a local ABC.

    Parameters
    ----------
    kind : str
        ``"Stacey"`` or ``"CE_A1"``.
    dt_scheme : str
        ``"backward"``, ``"backward_2nd"`` or ``"central"``.

    Returns
    -------
    dict
        The boundary condition entry of the model dictionary.
    """
    return {
        "status": True,
        "abc_type": "nrbc",
        "nrbc": {"type": kind, "dt_scheme": dt_scheme},
    }


def make_dictionary(
    material_parameters: dict,
    absorbing: dict | None = None,
    clamped: bool = False,
    dt: float = DT,
) -> dict:
    """Build a compact two-dimensional elastic model.

    Parameters
    ----------
    material_parameters : dict
        Complete Lame or velocity material parameterization.
    absorbing : dict, optional
        ``absorving_boundary_conditions`` entry, from :func:`local_abc`.
        ``None`` leaves the boundary traction-free.
    clamped : bool, optional
        Whether to clamp the displacement on every side with a homogeneous
        Dirichlet condition.
    dt : float, optional
        Time step.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    dictionary = {
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
            "final_time": FINAL_TIME,
            "dt": dt,
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
    if absorbing is not None:
        dictionary["absorving_boundary_conditions"] = absorbing
    if clamped:
        dictionary["boundary_conditions"] = [
            ("u", side, fire.Constant((0.0, 0.0))) for side in SIDES
        ]
    return dictionary


def build_wave(
    material_parameters: dict,
    absorbing: dict | None = None,
    clamped: bool = False,
    dt: float = DT,
    solver_parameters: dict | None = None,
) -> spyro.IsotropicWave:
    """Build a wave on the non-periodic mesh shared by every test here.

    Parameters
    ----------
    material_parameters : dict
        Complete Lame or velocity material parameterization.
    absorbing : dict, optional
        Passed on to :func:`make_dictionary`.
    clamped : bool, optional
        Passed on to :func:`make_dictionary`.
    dt : float, optional
        Passed on to :func:`make_dictionary`.
    solver_parameters : dict, optional
        PETSc options replacing spyro's default solve.

    Returns
    -------
    spyro.IsotropicWave
        The wave object, with mesh set, not yet solved.
    """
    wave = spyro.IsotropicWave(
        make_dictionary(material_parameters, absorbing, clamped, dt),
    )
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    if solver_parameters is not None:
        wave.set_solver_parameters(solver_parameters)
    return wave


@pytest.fixture(scope="module")
def free_surface_guess_record() -> np.ndarray:
    """Guess-model record with the traction-free boundary.

    Module scoped: it is the reference every other case is compared against
    to show its boundary condition actually acted on the solution.

    Returns
    -------
    numpy.ndarray
        Receiver time series of the guess model with a free boundary.
    """
    wave = build_wave(LAME_MATERIAL)
    wave.forward_solve()
    return wave.forward_solution_receivers


# Each case: guess material, boundary condition and, where the defaults do
# not do, the time step and the solver.
CASES = [
    pytest.param({"material": LAME_MATERIAL}, id="free-surface"),
    pytest.param({"material": LAME_MATERIAL, "clamped": True}, id="dirichlet"),
    pytest.param(
        {"material": LAME_MATERIAL, "absorbing": local_abc("Stacey", "backward")},
        id="stacey-backward",
    ),
    # The second-order backward derivative weighs the explicit boundary
    # term by 3/(2 dt) instead of 1/dt, which tightens the stable time step:
    # at DT this guess material blows up before the wave reaches the
    # receivers.
    pytest.param(
        {
            "material": LAME_MATERIAL,
            "absorbing": local_abc("Stacey", "backward_2nd"),
            "dt": 0.001,
        },
        id="stacey-backward_2nd",
    ),
    # The central derivative puts the boundary term on the left-hand side,
    # integrated with the default facet quadrature, so the operator is no
    # longer diagonal. spyro's default solve (one Jacobi sweep) then
    # computes diag(A)^-1 b while the tape differentiates A u = b, and the
    # Taylor test converges at first order (~1.0) although the tape is right,
    # as the exact-solve case below shows.
    pytest.param(
        {"material": LAME_MATERIAL, "absorbing": local_abc("Stacey", "central")},
        id="stacey-central",
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="the central scheme's boundary term makes the operator "
            "non-diagonal, which spyro's Jacobi solve does not invert",
        ),
    ),
    pytest.param(
        {
            "material": LAME_MATERIAL,
            "absorbing": local_abc("Stacey", "central"),
            "solver_parameters": EXACT_SOLVE,
        },
        id="stacey-central-exact-solve",
    ),
    pytest.param(
        {"material": LAME_MATERIAL, "absorbing": local_abc("CE_A1", "backward")},
        id="clayton-engquist-backward",
    ),
    # The absorbing terms are written in the wave speeds, so with this
    # parameterization they depend on the controls directly rather than
    # through the Lame parameters.
    pytest.param(
        {
            "material": VELOCITY_MATERIAL,
            "absorbing": local_abc("Stacey", "backward"),
        },
        id="stacey-backward-velocity",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_elastic_automated_adjoint_boundaries(free_surface_guess_record, case):
    """Taylor-test every control under the given boundary condition."""
    absorbing = case.get("absorbing")
    clamped = case.get("clamped", False)
    dt = case.get("dt", DT)
    solver_parameters = case.get("solver_parameters")

    wave = build_wave(
        case["material"], absorbing, clamped, dt, solver_parameters,
    )
    exact = build_wave(EXACT_MATERIAL, absorbing, clamped, dt, solver_parameters)
    exact.forward_solve()
    wave.real_shot_record = exact.forward_solution_receivers
    wave.enable_automated_adjoint()

    try:
        wave.forward_solve()

        assert isinstance(wave.automated_adjoint._tape, Tape)
        assert isinstance(wave.functional_value, AdjFloat)
        if (absorbing is not None or clamped) and dt == DT:
            # A Taylor test only says something about the boundary condition
            # if it changed the solution the functional is built from.
            assert not np.allclose(
                wave.forward_solution_receivers, free_surface_guess_record,
            ), "the boundary condition did not change the receiver data"

        gradients = wave.gradient_solve()
        assert all(
            isinstance(gradient, fire.Function)
            for gradient in gradients.values()
        )

        # Perturb by a tenth of each control to stay in the asymptotic
        # regime of the Taylor test (see test_elastic_auto_adj_2d.py).
        rng = np.random.default_rng(42)
        directions = [
            fire.Function(
                control.function_space(),
                val=0.1 * control.dat.data_ro * rng.random(
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
            "Elastic Taylor convergence rate %.4f < 1.90."
            % convergence_rate
        )
    finally:
        wave.automated_adjoint.clear_tape()


if __name__ == "__main__":
    pytest.main([__file__])
