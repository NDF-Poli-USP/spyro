"""Irksome time integration of the acoustic and elastic forward solves.

``time_axis["time_integration_scheme"] = "irksome"`` replaces the
central-difference loop by a Runge-Kutta-Nystrom stepper from Irksome, see
:mod:`spyro.solvers.time_integration_irksome`. The checks are:

* the options are validated and the tableau catalogue resolves names to the
  three stepper families;
* manufactured solutions, linear in time, are reproduced to round-off by
  every family, which exercises the Dirichlet conditions, the UFL sources
  and the initial conditions of both solvers;
* point-source shot records agree with the central-difference scheme to the
  latter's temporal error, with both source paths and with absorbing
  boundaries, and fourth-order tableaux agree with each other far better;
* the automated adjoint records the stepper, so the gradient passes a
  Taylor test;
* what the scheme does not do -- the PML and the implemented adjoint -- is
  refused with a clear error.
"""

import math
from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest
from firedrake import as_vector, errornorm

import spyro
from spyro.io.time_io import IrksomeOptions
from spyro.solvers import time_integration_irksome as tii
from spyro.utils.typing import AdjointType, TimeIntegrationScheme

from .model import dictionary as mms_model


pytestmark = pytest.mark.skipif(
    tii.irksome is None, reason="Irksome is not installed",
)

# One tableau of each stepper family, as (name, stages).
FAMILIES = [
    pytest.param("rk4", None, id="explicit-rk4"),
    pytest.param("alexander", None, id="dirk-alexander"),
    pytest.param("gauss_legendre", 2, id="implicit-gauss-legendre-2"),
]


def irksome_time_axis(time_axis: dict, tableau, stages=None) -> dict:
    """Return ``time_axis`` switched to the Irksome scheme.

    Parameters
    ----------
    time_axis : dict
        The ``time_axis`` entry of a model dictionary.
    tableau : str or object
        The tableau to select.
    stages : int, optional
        Number of stages for the collocation families.

    Returns
    -------
    dict
        A copy of ``time_axis`` with the scheme and its options set.
    """
    time_axis = deepcopy(time_axis)
    time_axis["time_integration_scheme"] = "irksome"
    time_axis["irksome"] = {"tableau": tableau, "stages": stages}
    return time_axis


# ---------------------------------------------------------------------------
# Options and tableaux
# ---------------------------------------------------------------------------


def test_options_defaults_and_validation():
    """The ``time_axis["irksome"]`` entry is validated when it is read."""
    options = IrksomeOptions.from_dictionary({})
    assert options.tableau == "rk4"
    assert options.stages is None
    assert options.bc_type is None
    assert options.solver_parameters is None

    with pytest.raises(ValueError, match="Unknown time_axis"):
        IrksomeOptions.from_dictionary({"tableaux": "rk4"})
    with pytest.raises(ValueError, match="stages"):
        IrksomeOptions.from_dictionary({"stages": 0})
    with pytest.raises(ValueError, match="bc_type"):
        IrksomeOptions.from_dictionary({"bc_type": "strong"})
    with pytest.raises(ValueError, match="solver_parameters"):
        IrksomeOptions.from_dictionary({"solver_parameters": "lu"})


def test_scheme_is_validated_by_the_model():
    """Unknown schemes are rejected, known ones become the enum."""
    dictionary = deepcopy(mms_model)
    dictionary["time_axis"]["time_integration_scheme"] = "leapfrog"
    with pytest.raises(ValueError):
        spyro.AcousticWave(dictionary=dictionary)

    dictionary = deepcopy(mms_model)
    dictionary["time_axis"]["time_integration_scheme"] = "irksome"
    dictionary["time_axis"]["irksome"] = {"tableau": "alexander"}
    wave = spyro.AcousticWave(dictionary=dictionary)
    assert wave.time_integrator is TimeIntegrationScheme.IRKSOME
    assert wave.uses_irksome
    assert wave.irksome_options.tableau == "alexander"

    wave = spyro.AcousticWave(dictionary=deepcopy(mms_model))
    assert wave.time_integrator is TimeIntegrationScheme.CENTRAL_DIFFERENCE
    assert not wave.uses_irksome


@pytest.mark.parametrize(
    ("name", "stages", "family", "num_stages"),
    [
        ("rk4", None, tii.EXPLICIT, 4),
        ("classic_nystrom4", None, tii.EXPLICIT, 4),
        ("backward_euler", None, tii.DIRK, 1),
        ("alexander", None, tii.DIRK, 3),
        ("qin_zhang", None, tii.DIRK, 2),
        # A one-stage collocation method is the implicit midpoint rule,
        # which is diagonally implicit.
        ("gauss_legendre", None, tii.DIRK, 1),
        ("gauss_legendre", 3, tii.FULLY_IMPLICIT, 3),
        ("radau_iia", 2, tii.FULLY_IMPLICIT, 2),
        # The two-stage Lobatto IIIA method is the trapezoidal rule, whose
        # Butcher matrix is lower triangular with an explicit first stage.
        ("lobatto_iiia", None, tii.DIRK, 2),
        ("lobatto_iiia", 3, tii.FULLY_IMPLICIT, 3),
        ("lobatto_iiic", 3, tii.FULLY_IMPLICIT, 3),
    ],
)
def test_tableau_catalogue(name, stages, family, num_stages):
    """Names resolve to tableaux of the expected family and size."""
    tableau = tii.make_tableau(IrksomeOptions(tableau=name, stages=stages))
    assert tableau.num_stages == num_stages
    assert tii.stepper_family(tableau) == family


def test_tableau_errors():
    """Wrong names, stage counts and objects are refused."""
    with pytest.raises(ValueError, match="Unknown Irksome tableau"):
        tii.make_tableau(IrksomeOptions(tableau="rk5"))
    with pytest.raises(ValueError, match="fixed number of stages"):
        tii.make_tableau(IrksomeOptions(tableau="rk4", stages=4))
    with pytest.raises(ValueError, match="at least 2 stages"):
        tii.make_tableau(IrksomeOptions(tableau="lobatto_iiic", stages=1))
    with pytest.raises(TypeError, match="ButcherTableau"):
        tii.make_tableau(IrksomeOptions(tableau=4))

    # Irksome tableau objects pass through untouched.
    tableau = tii.irksome.RadauIIA(2)
    assert tii.make_tableau(IrksomeOptions(tableau=tableau)) is tableau
    with pytest.raises(ValueError, match="tableau object"):
        tii.make_tableau(IrksomeOptions(tableau=tableau, stages=2))


def test_default_solver_parameters_follow_the_family():
    """Explicit stages get a forward substitution, implicit ones a solver.

    The forward substitution is a one-way sweep, which is not exact for the
    transposed system the adjoint solves, so enabling the automated adjoint
    switches it to the symmetric sweep.
    """
    wave = spyro.AcousticWave(dictionary=deepcopy(mms_model))
    explicit = tii.default_solver_parameters(wave, tii.EXPLICIT)
    assert explicit["pc_fieldsplit_type"] == "multiplicative"
    assert explicit["fieldsplit_pc_type"] == "jacobi"
    assert tii.default_solver_parameters(wave, tii.DIRK)["ksp_type"] == "cg"
    assert tii.default_solver_parameters(wave, tii.FULLY_IMPLICIT)["pc_type"] == "lu"

    wave.adjoint_type = AdjointType.AUTOMATED_ADJOINT
    explicit = tii.default_solver_parameters(wave, tii.EXPLICIT)
    assert explicit["pc_fieldsplit_type"] == "symmetric_multiplicative"


# ---------------------------------------------------------------------------
# Manufactured solutions: exact for every family
# ---------------------------------------------------------------------------


def acoustic_mms_error(tableau, stages=None, cell_type="triangles"):
    """Error of the acoustic manufactured solution after a short Irksome run.

    Parameters
    ----------
    tableau : str
        Tableau name.
    stages : int, optional
        Number of stages for the collocation families.
    cell_type : str, optional
        ``"triangles"`` or ``"quadrilaterals"``.

    Returns
    -------
    float
        L2 error against the manufactured solution at the final time.
    """
    dictionary = deepcopy(mms_model)
    dictionary["acquisition"]["source_type"] = "MMS"
    dictionary["options"]["cell_type"] = cell_type
    dictionary["options"]["variant"] = "lumped"
    dictionary["time_axis"]["final_time"] = 0.1
    dictionary["time_axis"] = irksome_time_axis(
        dictionary["time_axis"], tableau, stages,
    )
    wave = spyro.AcousticWaveMMS(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    wave.set_initial_velocity_model(expression="1 + sin(pi*-z)*sin(pi*x)")
    wave.forward_solve()
    # The stepper carries the true time of its state, so ``current_time``
    # is where the solution has to be compared.
    assert math.isclose(
        wave.current_time, wave.dt * (int(0.1 / wave.dt) + 1),
    )
    return errornorm(wave.u_n, wave.analytical_solution(wave.current_time))


@pytest.mark.parametrize(("tableau", "stages"), FAMILIES)
def test_acoustic_mms(tableau, stages):
    """The acoustic manufactured solution is reproduced to round-off."""
    assert acoustic_mms_error(tableau, stages) < 1e-12


def test_acoustic_mms_quadrilaterals():
    """The spectral quadrilateral elements work the same way."""
    assert acoustic_mms_error("classic_nystrom4", cell_type="quadrilaterals") < 1e-12


def elastic_mms_error(tableau, stages=None):
    """Error of the elastic manufactured solution after a short Irksome run.

    Parameters
    ----------
    tableau : str
        Tableau name.
    stages : int, optional
        Number of stages for the collocation families.

    Returns
    -------
    float
        Largest L2 error of the displacement components at the final time.
    """
    u1 = lambda x, t: (x[0]**2 + x[0])*(x[1]**2 - x[1])*t
    u2 = lambda x, t: (2*x[0]**2 + 2*x[0])*(-x[1]**2 + x[1])*t
    u = lambda x, t: as_vector([u1(x, t), u2(x, t)])

    b1 = lambda x, t: -(2*x[0]**2 + 6*x[1]**2 - 16*x[0]*x[1] + 10*x[0] - 14*x[1] + 4)*t
    b2 = lambda x, t: -(-12*x[0]**2 - 4*x[1]**2 + 8*x[0]*x[1] - 16*x[0] + 8*x[1] - 2)*t
    b = lambda x, t: as_vector([b1(x, t), b2(x, t)])

    dictionary = deepcopy(mms_model)
    dictionary["acquisition"]["source_type"] = "MMS"
    dictionary["acquisition"]["body_forces"] = b
    dictionary["time_axis"]["initial_condition"] = u
    dictionary["time_axis"]["dt"] = 1e-3
    dictionary["time_axis"]["final_time"] = 0.1
    dictionary["time_axis"]["output_frequency"] = 100
    dictionary["time_axis"] = irksome_time_axis(
        dictionary["time_axis"], tableau, stages,
    )
    dictionary["synthetic_data"] = {
        "type": "object",
        "density": 1,
        "lambda": 1,
        "mu": 1,
        "real_velocity_file": None,
    }
    dictionary["boundary_conditions"] = [
        ("uz", "on_boundary", 0),
        ("ux", "on_boundary", 0),
    ]
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    wave.forward_solve()

    u_an = fire.Function(wave.function_space)
    u_an.interpolate(u(wave.get_spatial_coordinates(), wave.current_time))
    return max(
        errornorm(wave.u_n.sub(0), u_an.sub(0)),
        errornorm(wave.u_n.sub(1), u_an.sub(1)),
    )


@pytest.mark.parametrize(("tableau", "stages"), FAMILIES)
def test_elastic_mms(tableau, stages):
    """The elastic manufactured solution is reproduced to round-off.

    It has Dirichlet conditions on every component, time-dependent body
    forces and an initial condition with a non-zero velocity.
    """
    assert elastic_mms_error(tableau, stages) < 1e-12


# ---------------------------------------------------------------------------
# Point sources: agreement with the central-difference scheme
# ---------------------------------------------------------------------------


def shot_dictionary(scheme="central_difference", tableau="rk4", stages=None,
                    absorbing=None, use_vertex_only_mesh=False, elastic=False,
                    final_time=0.6):
    """Build a small two-dimensional point-source model.

    Parameters
    ----------
    scheme : str, optional
        Time integration scheme.
    tableau : str, optional
        Irksome tableau, used when ``scheme`` is ``"irksome"``.
    stages : int, optional
        Number of stages for the collocation families.
    absorbing : dict, optional
        ``absorving_boundary_conditions`` entry.
    use_vertex_only_mesh : bool, optional
        Which point-source and receiver path to use.
    elastic : bool, optional
        Whether to build an isotropic elastic model.
    final_time : float, optional
        Final time of the simulation.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    dictionary = {
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
            "source_locations": [(-0.3, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect(
                (-0.7, 0.2), (-0.7, 0.8), 5
            ),
            "use_vertex_only_mesh": use_vertex_only_mesh,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": final_time,
            "dt": 1e-3,
            "output_frequency": 10**6,
            "gradient_sampling_frequency": 1,
            "time_integration_scheme": scheme,
            "irksome": {"tableau": tableau, "stages": stages},
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }
    if absorbing is not None:
        dictionary["absorving_boundary_conditions"] = absorbing
    if elastic:
        dictionary["acquisition"]["amplitude"] = np.array([0.0, 1.0])
        dictionary["synthetic_data"] = {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": 1.5,
            "s_wave_velocity": 0.9,
            "real_velocity_file": None,
        }
    return dictionary


def shot_record(**kwargs):
    """Run the model of :func:`shot_dictionary` and return its receiver data.

    Parameters
    ----------
    **kwargs
        Passed on to :func:`shot_dictionary`.

    Returns
    -------
    numpy.ndarray
        Receiver time series of the single shot.
    """
    dictionary = shot_dictionary(**kwargs)
    if kwargs.get("elastic", False):
        wave = spyro.IsotropicWave(dictionary)
    else:
        wave = spyro.AcousticWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    if not kwargs.get("elastic", False):
        wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    return np.asarray(wave.forward_solution_receivers)


def relative_difference(record, reference):
    """Relative L2 distance between two shot records.

    Parameters
    ----------
    record : numpy.ndarray
        Record under test.
    reference : numpy.ndarray
        Record it is compared to.

    Returns
    -------
    float
        ``||record - reference|| / ||reference||``.
    """
    return np.linalg.norm(record - reference) / np.linalg.norm(reference)


@pytest.fixture(scope="module")
def acoustic_reference():
    """Acoustic records of the central-difference scheme and of RK4."""
    return {
        "central_difference": shot_record(),
        "rk4": shot_record(scheme="irksome", tableau="rk4"),
    }


def test_acoustic_shot_matches_central_difference(acoustic_reference):
    """RK4 and central differences agree to the latter's temporal error.

    The records index the same time levels, so nothing has to be shifted.
    """
    rk4 = acoustic_reference["rk4"]
    central = acoustic_reference["central_difference"]
    assert rk4.shape == central.shape
    assert np.abs(rk4).max() > 1e-3, "the wave never reached the receivers"
    assert relative_difference(rk4, central) < 5e-3


@pytest.mark.parametrize(
    ("tableau", "stages", "tolerance"),
    [
        pytest.param("classic_nystrom4", None, 1e-5, id="classic_nystrom4"),
        pytest.param("gauss_legendre", 2, 1e-5, id="gauss_legendre_2"),
        pytest.param("alexander", None, 1e-3, id="alexander"),
    ],
)
def test_acoustic_tableaux_agree(acoustic_reference, tableau, stages, tolerance):
    """Higher-order tableaux agree with RK4 far better than central differences do."""
    record = shot_record(scheme="irksome", tableau=tableau, stages=stages)
    assert relative_difference(record, acoustic_reference["rk4"]) < tolerance


def test_both_source_paths_give_the_same_shot(acoustic_reference):
    """The vertex-only-mesh and the tabulated point sources coincide."""
    record = shot_record(
        scheme="irksome", tableau="rk4", use_vertex_only_mesh=True,
    )
    assert relative_difference(record, acoustic_reference["rk4"]) < 1e-10


def test_nrbc_shot_matches_central_difference():
    """The non-reflecting boundary term acts on the stage velocities."""
    absorbing = {
        "status": True,
        "abc_type": "nrbc",
        "absorb_top": True,
        "absorb_bottom": True,
        "absorb_right": True,
        "absorb_left": True,
    }
    rk4 = shot_record(
        scheme="irksome", tableau="rk4", absorbing=absorbing, final_time=0.8,
    )
    central = shot_record(absorbing=absorbing, final_time=0.8)
    reflecting = shot_record(scheme="irksome", tableau="rk4", final_time=0.8)
    assert relative_difference(rk4, reflecting) > 1e-2, \
        "the NRBC did not change the receiver data"
    assert relative_difference(rk4, central) < 5e-3


def test_elastic_shot_matches_central_difference():
    """The elastic point source and the Stacey boundary follow the scheme."""
    absorbing = {
        "status": True,
        "abc_type": "nrbc",
        "nrbc": {"type": "Stacey", "dt_scheme": "backward"},
    }
    rk4 = shot_record(scheme="irksome", tableau="rk4", elastic=True, final_time=0.5)
    central = shot_record(elastic=True, final_time=0.5)
    assert rk4.shape == central.shape
    assert np.abs(rk4).max() > 1e-3, "the wave never reached the receivers"
    assert relative_difference(rk4, central) < 5e-3

    rk4_abc = shot_record(
        scheme="irksome", tableau="rk4", elastic=True, absorbing=absorbing,
        final_time=0.5,
    )
    central_abc = shot_record(elastic=True, absorbing=absorbing, final_time=0.5)
    assert relative_difference(rk4_abc, rk4) > 1e-4, \
        "the local ABC did not change the receiver data"
    assert relative_difference(rk4_abc, central_abc) < 5e-3


# ---------------------------------------------------------------------------
# Automated adjoint
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.parametrize(("tableau", "stages"), FAMILIES)
def test_taylor_test_automated_adjoint(tableau, stages):
    """The stepper is recorded on the tape, so the gradient is exact.

    The time enters the stage forms through the source wavelet, and the
    tape only replays it because it is a coefficient: the test would fail
    with a ``Constant`` time, whose value the replay never updates.
    """
    absorbing = {
        "status": True,
        "abc_type": "nrbc",
        "absorb_bottom": True,
        "absorb_right": True,
        "absorb_left": True,
    }
    observed = shot_dictionary(
        scheme="irksome", tableau=tableau, stages=stages,
        absorbing=absorbing, final_time=0.4,
    )
    wave = spyro.AcousticWave(dictionary=observed)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.0),
        dg_velocity_model=False,
    )
    wave.forward_solve()
    real_shot_record = wave.forward_solution_receivers

    wave = spyro.AcousticWave(dictionary=deepcopy(observed))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.set_initial_velocity_model(constant=2.0)
    wave.real_shot_record = real_shot_record
    wave.enable_automated_adjoint()
    try:
        wave.forward_solve()
        dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
        assert isinstance(dJ, fire.Function)
        assert fire.norm(dJ) > 0.0

        size, = np.shape(wave.c.dat.data_ro[:])
        direction = fire.Function(
            wave.c.function_space(),
            val=np.random.default_rng(0).random(size),
        )
        rate = wave.automated_adjoint.verify_gradient(
            wave.c, direction=direction, dJdm=dJ,
        )
        assert rate > 1.9, f"Taylor convergence rate {rate} with {tableau}"
    finally:
        wave.automated_adjoint.clear_tape()


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_gradient_matches_with_checkpointing():
    """A checkpoint schedule recomputes the stepper and gets the same gradient.

    The mixed schedule keeps a few checkpoints and replays the forward steps
    between them, so the time steps have to be closed on the tape where the
    stepper ends them.
    """
    absorbing = {"status": True, "abc_type": "nrbc", "absorb_bottom": True}
    dictionary = shot_dictionary(
        scheme="irksome", tableau="rk4", absorbing=absorbing, final_time=0.4,
    )
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.0),
        dg_velocity_model=False,
    )
    wave.forward_solve()
    real_shot_record = wave.forward_solution_receivers

    gradients = {}
    for name, kwargs in {
        "none": {"checkpointing": False, "snapshots": None},
        "mixed": {"checkpointing": True, "snapshots": 10},
    }.items():
        wave = spyro.AcousticWave(dictionary=deepcopy(dictionary))
        wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
        wave.set_initial_velocity_model(constant=2.0)
        wave.real_shot_record = real_shot_record
        wave.enable_automated_adjoint(**kwargs)
        try:
            wave.forward_solve()
            dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
            gradients[name] = dJ.copy(deepcopy=True)
        finally:
            wave.automated_adjoint.clear_tape()

    reference = gradients["none"].dat.data_ro
    assert np.linalg.norm(reference) > 0.0
    assert np.allclose(
        gradients["mixed"].dat.data_ro, reference,
        rtol=1e-10, atol=1e-12 * np.abs(reference).max(),
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_taylor_test_elastic_automated_adjoint():
    """The elastic material derivatives are exact under the stepper too."""
    dictionary = shot_dictionary(
        scheme="irksome", tableau="rk4", elastic=True, final_time=0.4,
    )
    dictionary["synthetic_data"] = {
        "type": "object",
        "density": 1.0,
        "lambda": 1.2,
        "mu": 0.8,
        "real_velocity_file": None,
    }
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.forward_solve()
    real_shot_record = wave.forward_solution_receivers

    dictionary = deepcopy(dictionary)
    dictionary["synthetic_data"].update({"lambda": 1.0, "mu": 1.0})
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.real_shot_record = real_shot_record
    wave.enable_automated_adjoint()
    try:
        wave.forward_solve()
        gradients = wave.gradient_solve()
        controls = wave.automated_adjoint.controls
        assert len(gradients) == len(controls) == 3
        rng = np.random.default_rng(42)
        directions = [
            fire.Function(
                control.function_space(),
                val=0.1 * control.dat.data_ro * rng.random(
                    control.dat.data_ro.shape,
                ),
            )
            for control in controls
        ]
        rate = wave.automated_adjoint.verify_gradient(
            controls, direction=directions, dJdm=gradients,
        )
        assert rate > 1.9, f"Elastic Taylor convergence rate {rate}"
    finally:
        wave.automated_adjoint.clear_tape()


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_fwi_automated_adjoint(tmp_path, monkeypatch):
    """A whole inversion runs on the stepper: data, tape, TAO and result."""
    from .test_fwi_automated_adjoint import (
        ACOUSTIC_GUESS, ACOUSTIC_REAL, build_dictionary,
    )

    vmin, vmax = 2.0, 3.5
    monkeypatch.chdir(tmp_path)
    dictionary = build_dictionary()
    dictionary["time_axis"] = irksome_time_axis(
        dictionary["time_axis"], "rk4",
    )

    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(ACOUSTIC_REAL)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)
    assert fwi.wave.uses_irksome

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    assert isinstance(result, fire.Function)
    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, ACOUSTIC_GUESS)
    assert len(fwi.functional_history) > 1
    assert fwi.functional_history[-1] < fwi.functional_history[0]


# ---------------------------------------------------------------------------
# What the scheme refuses
# ---------------------------------------------------------------------------


def test_pml_is_refused():
    """The PML auxiliary equation is first order in time: no Nystrom form."""
    absorbing = {
        "status": True,
        "abc_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    dictionary = shot_dictionary(
        scheme="irksome", tableau="rk4", absorbing=absorbing,
    )
    wave = spyro.AcousticWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    with pytest.raises(NotImplementedError, match="PML"):
        wave.forward_solve()


def test_implemented_adjoint_is_refused():
    """The hand-written adjoint integrates central differences backwards."""
    dictionary = shot_dictionary(scheme="irksome", tableau="rk4")
    wave = spyro.AcousticWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    wave.real_shot_record = wave.forward_solution_receivers
    with pytest.raises(NotImplementedError, match="AUTOMATED_ADJOINT"):
        wave.gradient_solve(adjoint_type=AdjointType.IMPLEMENTED_ADJOINT)
