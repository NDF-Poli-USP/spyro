"""Time-step estimates from the stiffness and mass forms of each wave physics."""
import warnings

import numpy as np
import pytest
from firedrake import (
    Constant, Function, FunctionSpace, TestFunction, TrialFunction,
    UnitSquareMesh, dx, grad, inner,
)

import spyro
from spyro.solvers.modal.modal_forms_and_matrices import (
    maximum_diagonal_ratio,
    acoustic_weak_forms_c_in_stiffness,
)
from spyro.solvers.modal.modal_sol import Modal_Solver


def _elastic_wave(vp: float, vs: float) -> spyro.IsotropicWave:
    """Build a small homogeneous 2D isotropic elastic wave.

    Parameters
    ----------
    vp : float
        P-wave velocity.
    vs : float
        S-wave velocity.

    Returns
    -------
    spyro.IsotropicWave
        Wave with its mesh and material parameters set.
    """
    dictionary = {
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
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.5, 0.5)],
            "frequency": 5.0,
            "receiver_locations": [(-0.6, 0.6)],
            "amplitude": np.array([0.0, 1.0]),
        },
        "time_axis": {
            # Long enough that rounding to a whole number of steps is small
            "final_time": 20.0,
            "dt": 0.001,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": vp,
            "s_wave_velocity": vs,
            "real_velocity_file": None,
        },
    }
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave._initialize_model_parameters()
    return wave


@pytest.mark.parametrize("method", ["ANALYTICAL", "ARNOLDI"])
def test_forms_match_default_acoustic_forms(method: str):
    """Passing the default acoustic forms gives the default estimate."""
    mesh = UnitSquareMesh(6, 6, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = mesh.coordinates
    c = Function(V).interpolate(1.5 + x + y)
    solver = Modal_Solver(2, method=method, calc_max_dt=True)

    with pytest.warns(DeprecationWarning, match="without forms"):
        default = solver.estimate_timestep(c, V, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with_forms = solver.estimate_timestep(
            c, V, 1.0, forms=acoustic_weak_forms_c_in_stiffness(c, V)
        )
    assert np.isclose(default, with_forms)


@pytest.mark.parametrize("estimate_max_eigenvalue", [True, False])
def test_elastic_timestep_scales_with_velocity(estimate_max_eigenvalue: bool):
    """Doubling the elastic velocities halves the estimated time step."""
    dt = _elastic_wave(1.5, 1.0).get_and_set_maximum_dt(
        fraction=1.0, estimate_max_eigenvalue=estimate_max_eigenvalue
    )
    doubled = _elastic_wave(3.0, 2.0).get_and_set_maximum_dt(
        fraction=1.0, estimate_max_eigenvalue=estimate_max_eigenvalue
    )
    assert dt > 0.0
    assert np.isclose(doubled, dt / 2, rtol=1e-2)


def test_elastic_analytical_timestep_uses_elastic_forms():
    """The elastic estimate comes from the elastic stiffness and mass forms."""
    wave = _elastic_wave(1.5, 1.0)
    max_eigval = maximum_diagonal_ratio(*wave.modal_weak_forms())
    with warnings.catch_warnings():
        # The wave passes its own forms, so the deprecated path is unused
        warnings.simplefilter("error", DeprecationWarning)
        dt = wave.get_and_set_maximum_dt(
            fraction=1.0, estimate_max_eigenvalue=True
        )
    assert np.isclose(dt, 2.0 / np.sqrt(max_eigval), rtol=1e-2)
    assert wave.dt == dt


def test_maximum_diagonal_ratio_rejects_zero_stiffness():
    """A zero stiffness gives no finite time step, so it is rejected."""
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = Constant(0.0)*inner(grad(u), grad(v))*dx
    m = u*v*dx
    with pytest.raises(ValueError, match="no nonzero ratio"):
        maximum_diagonal_ratio(a, m)


@pytest.mark.parametrize(
    "method", ["KRYLOVSCH_CH", "KRYLOVSCH_CG", "KRYLOVSCH_GH", "KRYLOVSCH_GG"]
)
def test_slepc_timestep_matches_scipy(method: str):
    """SLEPc finds the same largest eigenvalue as SciPy's ARNOLDI."""
    mesh = UnitSquareMesh(6, 6, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = mesh.coordinates
    c = Function(V).interpolate(1.5 + x + y)
    forms = acoustic_weak_forms_c_in_stiffness(c, V)

    scipy_solver = Modal_Solver(2, method="ARNOLDI", calc_max_dt=True)
    slepc_solver = Modal_Solver(2, method=method, calc_max_dt=True)
    scipy_dt = scipy_solver.estimate_timestep(c, V, 1.0, forms=forms)
    slepc_dt = slepc_solver.estimate_timestep(c, V, 1.0, forms=forms)
    assert np.isclose(slepc_dt, scipy_dt, rtol=1e-6)


def test_timestep_methods():
    """SLEPc methods can estimate the time step, the Rayleigh quotient not."""
    assert Modal_Solver(calc_max_dt=True).method == "KRYLOVSCH_CH"
    with pytest.raises(ValueError):
        Modal_Solver(method="RAYLEIGH", calc_max_dt=True)
