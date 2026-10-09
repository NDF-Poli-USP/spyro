"""Physical regularization before coordinate transformations."""

from types import SimpleNamespace

import firedrake as fire
import firedrake.adjoint as adj
import numpy as np
import pytest
from pyadjoint import Tape

from spyro import ElasticMaterialParameter as P
from spyro.functionals import H1Regularization, InversionObjective, L2DataMisfit
from spyro.functionals.reduced import (
    LatentReducedFunctional, LumpedL2ReducedFunctional,
)
from spyro.utils.typing import FunctionalEvaluationMode as Mode
from spyro.domains.space import create_function_space

pytestmark = pytest.mark.newer_firedrake


@pytest.fixture(autouse=True)
def fresh_tape():
    """Isolate annotation between tests.

    Yields
    ------
    Tape
        Fresh working tape.
    """
    tape = Tape()
    adj.set_working_tape(tape)
    adj.continue_annotation()
    yield tape
    adj.pause_annotation()
    tape.clear_tape()


def test_h1_value_reference_scale_and_zero() -> None:
    """Check exact affine-field integrals and disabled regularization."""
    mesh = fire.UnitSquareMesh(3, 3)
    space = fire.FunctionSpace(mesh, "CG", 1)
    x, y = fire.SpatialCoordinate(mesh)
    vp = fire.Function(space).interpolate(x + 2 * y)
    vs = fire.Function(space).interpolate(y)
    reg = H1Regularization({P.P_WAVE_VELOCITY: 2, P.S_WAVE_VELOCITY: 4},
                           scales={P.P_WAVE_VELOCITY: 2})
    fields = {P.P_WAVE_VELOCITY: vp, P.S_WAVE_VELOCITY: vs}
    assert float(reg(fields)) == pytest.approx(3.25)
    ref = fire.Function(space).assign(vp)
    assert float(H1Regularization({P.P_WAVE_VELOCITY: 1},
                                  references={P.P_WAVE_VELOCITY: ref})(fields)) == 0
    assert H1Regularization({P.P_WAVE_VELOCITY: 0})(fields) == 0
    assert float(reg({P.P_WAVE_VELOCITY: fire.Function(space).assign(2),
                      P.S_WAVE_VELOCITY: fire.Function(space).assign(1)})) == 0


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weight(weight: float) -> None:
    """Reject invalid weights.

    Parameters
    ----------
    weight : float
        Invalid penalty coefficient.
    """
    with pytest.raises(ValueError):
        H1Regularization({P.P_WAVE_VELOCITY: weight})


def test_discontinuous_controls_rejected() -> None:
    """Do not silently interpret DG0 gradients as useful H1 penalties."""
    space = fire.FunctionSpace(fire.UnitSquareMesh(2, 2), "DG", 0)
    with pytest.raises(ValueError, match="continuous"):
        H1Regularization({P.P_WAVE_VELOCITY: 1})(
            {P.P_WAVE_VELOCITY: fire.Function(space).assign(1)},
        )


def test_reference_is_frozen() -> None:
    """A reference taken from the control is a fixed snapshot, not an alias."""
    mesh = fire.UnitSquareMesh(2, 2)
    space = fire.FunctionSpace(mesh, "CG", 1)
    x = fire.SpatialCoordinate(mesh)[0]
    m = fire.Function(space).interpolate(x)
    reg = H1Regularization({P.P_WAVE_VELOCITY: 2},
                           references={P.P_WAVE_VELOCITY: m})
    m.interpolate(2 * x)
    control = adj.Control(m)
    value = reg({P.P_WAVE_VELOCITY: m})
    assert float(value) == pytest.approx(1)
    rf = adj.ReducedFunctional(value, control)
    assert adj.taylor_test(rf, m, fire.Function(space).interpolate(x)) > 1.9


@pytest.mark.parametrize("wrapper", ["physical", "lumped", "latent"])
def test_composed_objective_taylor(wrapper: str) -> None:
    """Differentiate H1 through the same coordinates as the data misfit.

    Parameters
    ----------
    wrapper : str
        Coordinate wrapper under test.
    """
    mesh = fire.UnitSquareMesh(3, 3)
    space = create_function_space(mesh, "mass_lumped_triangle", 2)
    x, y = fire.SpatialCoordinate(mesh)
    m = fire.Function(space).interpolate(1.5 + 0.1 * fire.sin(x + y))
    control = adj.Control(m)
    data = fire.assemble(0.5 * (m ** 2 - 2) ** 2 * fire.dx)
    objective = InversionObjective(regularization=H1Regularization({P.P_WAVE_VELOCITY: 0.3}))
    total = objective.local_value(data, {P.P_WAVE_VELOCITY: m})
    rf = adj.ReducedFunctional(total, control)
    if wrapper == "lumped":
        rf = LumpedL2ReducedFunctional(rf)
    elif wrapper == "latent":
        rf = LatentReducedFunctional(rf, bounds=[(1.0, 3.0)])
    start = rf.controls[0].control
    direction = fire.Function(space).interpolate(0.1 + x * y)
    assert adj.taylor_test(rf, start, direction) > 1.9


def test_local_ensemble_scaling() -> None:
    """The regularizer must occur once in the ensemble sum."""
    mesh = fire.UnitSquareMesh(2, 2)
    space = fire.FunctionSpace(mesh, "CG", 1)
    m = fire.Function(space).interpolate(fire.SpatialCoordinate(mesh)[0])
    objective = InversionObjective(regularization=H1Regularization({P.P_WAVE_VELOCITY: 2}))
    fields = {P.P_WAVE_VELOCITY: m}
    for members in (1, 2, 4):
        assert members * float(objective.local_value(3 / members, fields, members)) == pytest.approx(4)


@pytest.mark.parametrize("shape", [(5, 3), (5, 3, 2)])
def test_l2_temporal_consistency(shape: tuple) -> None:
    """Scalar and elastic vector traces use the same time integration.

    Parameters
    ----------
    shape : tuple
        Residual array dimensions.
    """
    wave = SimpleNamespace(dt=0.1, use_vertex_only_mesh=False)
    residual = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100
    misfit = L2DataMisfit()
    per_step = sum(misfit(wave, row, Mode.PER_TIMESTEP, i, 5)
                   for i, row in enumerate(residual))
    assert per_step == pytest.approx(misfit(wave, residual))


def test_compatibility_imports() -> None:
    """Legacy and unified imports resolve to the same classes."""
    from spyro import reduced_functionals as old
    assert old.LatentReducedFunctional is LatentReducedFunctional
    assert old.LumpedL2ReducedFunctional is LumpedL2ReducedFunctional


@pytest.mark.parametrize("snapshots", [None, 2])
def test_acoustic_regularized_fwi(tmp_path, monkeypatch, snapshots: int | None) -> None:
    """Exercise objective plumbing and Taylor convergence with checkpointing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Temporary working-directory manager.
    snapshots : int or None
        None disables checkpointing; two enables mixed checkpointing.
    """
    import spyro
    from tests.on_one_core.test_fwi_automated_adjoint import build_dictionary
    from spyro.utils.typing import AdjointType

    monkeypatch.chdir(tmp_path)
    dictionary = build_dictionary()
    dictionary["time_axis"]["final_time"] = 0.04
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(3.0)
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(expression="2.5 + 0.1*x", dg_velocity_model=False)
    key = spyro.AcousticMaterialParameter.P_WAVE_VELOCITY
    objective = InversionObjective(regularization=H1Regularization({key: 0.01}))
    fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        objective=objective, maxiter=2, vmin=2.0, vmax=3.5,
        save_controls=False,
        adjoint_options={"checkpointing": snapshots is not None, "snapshots": snapshots},
    )
    rf = fwi.wave.automated_adjoint.reduced_functional
    initial_data = L2DataMisfit()(
        fwi.wave, np.asarray(fwi.wave.real_shot_record) - fwi.wave.forward_solution_receivers,
    )
    assert fwi.functional_history[0] == pytest.approx(float(initial_data) + 5e-5)
    start = rf.controls[0].tape_value().copy(deepcopy=True)
    space = start.function_space()
    x = fire.SpatialCoordinate(space.mesh())[0]
    direction = fire.Function(space).interpolate(0.1 + x)
    assert adj.taylor_test(rf, start, direction) > 1.9
    assert fwi.functional_history[-1] <= fwi.functional_history[0]


@pytest.mark.parametrize("spatial_size", [1, 2])
def test_mpi_objective_and_gradient(spatial_size: int) -> None:
    """Check ensemble and spatial reductions against analytic integrals.

    Parameters
    ----------
    spatial_size : int
        MPI ranks per spatial mesh.
    """
    if fire.COMM_WORLD.size % spatial_size:
        pytest.skip("MPI size must be divisible by the spatial size.")
    ensemble = fire.Ensemble(fire.COMM_WORLD, spatial_size)
    mesh = fire.UnitSquareMesh(3, 3, comm=ensemble.comm)
    space = fire.FunctionSpace(mesh, "CG", 1)
    x = fire.SpatialCoordinate(mesh)[0]
    m = fire.Function(space).interpolate(1 + x)
    control = adj.Control(m)
    count = ensemble.ensemble_comm.size
    data = fire.assemble(0.5 * m ** 4 * fire.dx) / count
    objective = InversionObjective(regularization=H1Regularization({P.P_WAVE_VELOCITY: 2}))
    total = objective.local_value(data, {P.P_WAVE_VELOCITY: m}, count)
    rf = adj.EnsembleReducedFunctional(total, control, ensemble, scatter_control=True)
    assert float(rf(m)) == pytest.approx(4.1)
    gradient = rf.derivative()
    with gradient.dat.vec_ro as g, m.dat.vec_ro as direction:
        assert g.dot(direction) == pytest.approx(14.4)


def test_sem4_extruded_h1_taylor() -> None:
    """Verify continuous 3D SEM4 controls on extruded hexahedra."""
    base = fire.UnitSquareMesh(2, 2, quadrilateral=True)
    mesh = fire.ExtrudedMesh(base, layers=2, layer_height=0.5)
    space = create_function_space(mesh, "spectral_quadrilateral", 4)
    x, y, z = fire.SpatialCoordinate(mesh)
    m = fire.Function(space).interpolate(1.5 + 0.2 * x * y + z ** 2)
    control = adj.Control(m)
    value = H1Regularization({P.P_WAVE_VELOCITY: 1})({P.P_WAVE_VELOCITY: m})
    rf = adj.ReducedFunctional(value, control)
    direction = fire.Function(space).interpolate(0.1 + x + y * z)
    assert adj.taylor_test(rf, m, direction) > 1.9


def test_elastic_regularized_fwi(tmp_path, monkeypatch) -> None:
    """Differentiate both elastic velocities in a short regularized FWI.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Temporary working-directory manager.
    """
    import spyro
    from tests.on_one_core.test_fwi_automated_adjoint import (
        build_elastic_dictionary, ELASTIC_GUESS, ELASTIC_REAL,
    )
    from spyro.utils.typing import AdjointType

    monkeypatch.chdir(tmp_path)
    dictionary = build_elastic_dictionary(ELASTIC_GUESS)
    dictionary["time_axis"]["final_time"] = 0.04
    fwi = spyro.FullWaveformInversion(dictionary=dictionary, wave_class=spyro.IsotropicWave)
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({P(key): value for key, value in ELASTIC_REAL.items()})
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    objective = InversionObjective(regularization=H1Regularization(
        {P.P_WAVE_VELOCITY: 0.01, P.S_WAVE_VELOCITY: 0.02},
    ))
    fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT, objective=objective,
        maxiter=2, vmin=[1.0, 1.5, 0.5], vmax=[3.0, 4.0, 2.5], save_controls=False,
    )
    rf = fwi.wave.automated_adjoint.reduced_functional
    start = [c.tape_value().copy(deepcopy=True) for c in rf.controls]
    directions = [fire.Function(c.function_space()).interpolate(
        0.1 + fire.SpatialCoordinate(c.function_space().mesh())[0],
    ) for c in start]
    assert adj.taylor_test(rf, start, directions) > 1.9
    assert fwi.functional_history[-1] <= fwi.functional_history[0]
    full_value = rf(start)
    full_gradient = rf.derivative()
    driver = fwi.wave.automated_adjoint
    index = driver.control_parameter_names.index(P.P_WAVE_VELOCITY)
    partial = driver.create_partial_reduced_functional(
        fwi.wave.functional_value, [P.P_WAVE_VELOCITY],
    )
    assert float(partial(start[index])) == pytest.approx(float(full_value))
    assert np.allclose(partial.derivative().dat.data_ro, full_gradient[index].dat.data_ro)
    assert adj.taylor_test(partial, start[index], directions[index]) > 1.9


def test_auto_weight_settings() -> None:
    """``"auto"`` is a weight; other strings, bad fractions and unresolved calls fail."""
    mesh = fire.UnitSquareMesh(2, 2)
    space = fire.FunctionSpace(mesh, "CG", 1)
    vp = fire.Function(space).interpolate(fire.SpatialCoordinate(mesh)[0])
    reg = H1Regularization({P.P_WAVE_VELOCITY: "auto", P.S_WAVE_VELOCITY: 1.0})
    assert reg.automatic == [P.P_WAVE_VELOCITY]
    with pytest.raises(ValueError, match="still 'auto'"):
        reg({P.P_WAVE_VELOCITY: vp, P.S_WAVE_VELOCITY: vp})
    with pytest.raises(ValueError, match="'auto'"):
        H1Regularization({P.P_WAVE_VELOCITY: "fast"})
    with pytest.raises(ValueError, match="gradient_fraction"):
        H1Regularization({P.P_WAVE_VELOCITY: "auto"}, gradient_fraction=0.0)


def test_unit_derivative_is_the_penalty_derivative() -> None:
    """The unit-weight derivative matches a central difference of the penalty.

    The penalty is quadratic, so the central difference is exact.
    """
    mesh = fire.UnitSquareMesh(3, 3)
    space = fire.FunctionSpace(mesh, "CG", 2)
    x, y = fire.SpatialCoordinate(mesh)
    m = fire.Function(space).interpolate(x * y + x ** 2)
    h = fire.Function(space).interpolate(fire.sin(3 * x) * y)
    reg = H1Regularization({P.P_WAVE_VELOCITY: 1.0}, scales={P.P_WAVE_VELOCITY: 2.0})
    with adj.stop_annotating():
        derivative = reg.unit_derivative(P.P_WAVE_VELOCITY, m)
        plus = float(reg({P.P_WAVE_VELOCITY: fire.Function(space).assign(m + 0.1 * h)}))
        minus = float(reg({P.P_WAVE_VELOCITY: fire.Function(space).assign(m - 0.1 * h)}))
    assert fire.assemble(fire.action(derivative, h)) == pytest.approx((plus - minus) / 0.2)


def _acoustic_fwi(guess: str):
    """Return the acoustic inversion of ``test_acoustic_regularized_fwi``.

    Parameters
    ----------
    guess : str
        Expression of the starting velocity.

    Returns
    -------
    spyro.FullWaveformInversion
        Driver with observed data and the starting model.
    """
    import spyro
    from tests.on_one_core.test_fwi_automated_adjoint import build_dictionary

    dictionary = build_dictionary()
    dictionary["time_axis"]["final_time"] = 0.04
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(3.0)
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(expression=guess, dg_velocity_model=False)
    return fwi


def test_acoustic_auto_weight_fwi(tmp_path, monkeypatch) -> None:
    """An ``"auto"`` weight is chosen, and the logged parts add up to the objective.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Temporary working-directory manager.
    """
    import spyro
    from spyro.utils.typing import AdjointType

    monkeypatch.chdir(tmp_path)
    fwi = _acoustic_fwi("2.5 + 0.1*x")
    key = spyro.AcousticMaterialParameter.P_WAVE_VELOCITY
    regularization = H1Regularization({key: "auto"}, gradient_fraction=0.1)
    fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        objective=InversionObjective(regularization=regularization),
        maxiter=2, vmin=2.0, vmax=3.5, save_controls=False,
    )
    weight = regularization.weights[key]
    assert not regularization.automatic and weight > 0
    assert len(fwi.misfit_history) == len(fwi.functional_history)
    assert len(fwi.regularization_history) == len(fwi.functional_history)
    parts = np.add(fwi.misfit_history, fwi.regularization_history)
    assert np.allclose(parts, fwi.functional_history, rtol=1e-12)
    assert all(value > 0 for value in fwi.regularization_history)


def test_auto_weight_needs_a_nonuniform_start(tmp_path, monkeypatch) -> None:
    """A uniform start has no H1 gradient to size the weight against.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Temporary working-directory manager.
    """
    import spyro
    from spyro.utils.typing import AdjointType

    monkeypatch.chdir(tmp_path)
    fwi = _acoustic_fwi("2.5")
    key = spyro.AcousticMaterialParameter.P_WAVE_VELOCITY
    with pytest.raises(ValueError, match="uniform"):
        fwi.run_fwi(
            adjoint_type=AdjointType.AUTOMATED_ADJOINT,
            objective=InversionObjective(regularization=H1Regularization({key: "auto"})),
            maxiter=1, vmin=2.0, vmax=3.5, save_controls=False,
        )


def test_l_curve_corner() -> None:
    """The corner of an L, given out of order, is where it turns."""
    from spyro.tools.l_curve import l_curve_corner

    # log points (misfit, penalty) along increasing weights: down, then right.
    log_points = [(0.0, 4.0), (0.05, 2.0), (0.1, 0.2), (2.0, 0.1), (4.0, 0.05)]
    weights = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    order = [3, 0, 4, 2, 1]
    misfits = [np.exp(log_points[i][0]) for i in order]
    penalties = [np.exp(log_points[i][1]) for i in order]
    corner = l_curve_corner([weights[i] for i in order], misfits, penalties)
    assert [weights[i] for i in order][corner] == 1e-7
    with pytest.raises(ValueError, match="three"):
        l_curve_corner([1, 2], [1, 2], [2, 1])
    with pytest.raises(ValueError, match="positive"):
        l_curve_corner([1, 2, 3], [1, 0, 2], [3, 2, 1])


@pytest.mark.parametrize("continuation", [
    {"cycles": 1, "factor": 10}, {"cycles": 2, "factor": 1.0},
    {"cycles": 2}, {"cycles": 2.5, "factor": 10},
])
def test_invalid_continuation(continuation: dict) -> None:
    """Reject continuation settings before any solve.

    Parameters
    ----------
    continuation : dict
        Invalid settings.
    """
    import spyro

    with pytest.raises(ValueError):
        spyro.FullWaveformInversion._continuation(continuation)


def test_acoustic_continuation_fwi(tmp_path, monkeypatch) -> None:
    """Continuation divides the weights per cycle, and the parts still add up.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Temporary working-directory manager.
    """
    import spyro
    from spyro.utils.typing import AdjointType

    monkeypatch.chdir(tmp_path)
    fwi = _acoustic_fwi("2.5 + 0.1*x")
    key = spyro.AcousticMaterialParameter.P_WAVE_VELOCITY
    regularization = H1Regularization({key: 0.01})
    fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        objective=InversionObjective(regularization=regularization),
        continuation={"cycles": 2, "factor": 10}, maxiter=2,
        vmin=2.0, vmax=3.5, save_controls=False,
    )
    assert regularization.weights[key] == pytest.approx(0.001)
    assert fwi.current_iteration == 2
    parts = np.add(fwi.misfit_history, fwi.regularization_history)
    assert np.allclose(parts, fwi.functional_history, rtol=1e-12)
    with pytest.raises(ValueError, match="needs an objective"):
        fwi.run_fwi(adjoint_type=AdjointType.AUTOMATED_ADJOINT,
                    continuation={"cycles": 2, "factor": 10})
