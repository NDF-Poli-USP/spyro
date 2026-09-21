"""FWI runs driven by the automated adjoint, acoustic and isotropic elastic.

Each test is a whole synthetic experiment: a true model is propagated to make
the observed data, the inversion starts from a different model, and ``run_fwi``
has to move it towards the truth. What is under test is the plumbing between
the driver, the pyadjoint tape and PETSc TAO, so the models are constant and
the meshes coarse -- the reconstruction is not expected to be good, only to
improve, and to improve the parameters it was told to move.

The two media reach the optimizer differently, which is why both are run: an
acoustic inversion moves one control, its velocity model, while an isotropic
elastic one moves the three parameters its equation is written in, each on its
own scale and so with bounds of its own.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType


Parameter = spyro.ElasticMaterialParameter

# Constant models, differing enough to leave a residual worth descending.
ACOUSTIC_GUESS = 2.5
ACOUSTIC_REAL = 3.0
ELASTIC_GUESS = {
    "density": 2.0,
    "p_wave_velocity": 2.5,
    "s_wave_velocity": 1.2,
}
ELASTIC_REAL = {
    "density": 2.2,
    "p_wave_velocity": 3.0,
    "s_wave_velocity": 1.5,
}


def build_dictionary():
    """Return a one-source acoustic FWI configuration.

    Returns
    -------
    dict
        Model dictionary sized so the wave reaches the receivers -- otherwise
        the residual would be zero and there would be nothing to descend --
        while staying cheap enough for a handful of optimizer iterations.
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
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.4,
            "dt": 0.002,
            "amplitude": 1.0,
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


def build_elastic_dictionary(material):
    """Return the same configuration for an isotropic elastic medium.

    Parameters
    ----------
    material : dict
        Material the solver is built from, which is also the set the equation
        ends up written in terms of.

    Returns
    -------
    dict
        Model dictionary.
    """
    dictionary = build_dictionary()
    # A vector source, and the materials the elastic equation reads.
    dictionary["acquisition"]["amplitude"] = np.array([0.0, 1.0])
    dictionary["time_axis"].pop("amplitude")
    dictionary["synthetic_data"] = {
        "type": "object",
        **material,
        "real_velocity_file": None,
    }
    return dictionary


@pytest.mark.newer_firedrake
def test_fwi_automated_adjoint(tmp_path, monkeypatch):
    """Invert an acoustic velocity model.

    A fixed iteration budget is how FWI is normally run, and TAO reports that
    as a failure to converge; the driver has to hand back the last iterate
    rather than let the exception through.
    """
    vmin, vmax = 2.0, 3.5
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(ACOUSTIC_REAL)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    # TAO optimizes the control itself, so that is what comes back.
    assert isinstance(result, fire.Function)
    assert result.function_space() == fwi.wave.c.function_space()
    assert isinstance(
        fwi.wave.automated_adjoint.reduced_functional,
        fire_ad.EnsembleReducedFunctional,
    )

    # The bounds are respected, and the control actually moved.
    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, ACOUSTIC_GUESS)

    # The monitor logged the iterates on top of the recorded starting point,
    # and the run brought the functional down.
    assert len(fwi.functional_history) > 1
    assert fwi.functional_history[-1] < fwi.functional_history[0]
    assert fwi.functional == fwi.functional_history[-1]

    # The optimum is written back into the driver and the solver alike.
    assert np.allclose(fwi.control_parameter_result.dat.data_ro, values)
    assert np.allclose(fwi.wave.c.dat.data_ro, values)
    assert (tmp_path / "result.npy").exists()


@pytest.mark.newer_firedrake
@pytest.mark.parametrize("adjoint_type", [
    AdjointType.AUTOMATED_ADJOINT, AdjointType.IMPLEMENTED_ADJOINT,
])
def test_fwi_gradient_mask(tmp_path, monkeypatch, adjoint_type):
    """A gradient mask freezes the model where it is zero, on either path.

    The mask keeps the lower half of the domain and zeroes the upper one,
    where the source and the receivers sit: no update the optimizer makes
    reaches it, whether the optimizer is TAO on the automated adjoint or
    L-BFGS-B on the implemented one. The gradients the solver returns are
    not masked, which keeps them the derivative of the misfit; the ones the
    driver returns are.
    """
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(ACOUSTIC_REAL)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)
    keep_below = fire.conditional(fwi.wave.mesh_z < -0.5, 1.0, 0.0)

    result = fwi.run_fwi(
        adjoint_type=adjoint_type, gradient_mask=keep_below,
        vmin=2.0, vmax=3.5, maxiter=2,
    )
    if adjoint_type is AdjointType.IMPLEMENTED_ADJOINT:
        result = fwi.control_parameter_result

    mask = fire.Function(result.function_space()).interpolate(keep_below)
    frozen = mask.dat.data_ro < 0.5
    assert frozen.any() and (~frozen).any(), "the mask splits the domain"

    values = result.dat.data_ro
    assert np.allclose(values[frozen], ACOUSTIC_GUESS), (
        "the model moved where the gradient is masked"
    )
    assert not np.allclose(values[~frozen], ACOUSTIC_GUESS), (
        "the model did not move where the gradient is kept"
    )
    assert fwi.functional_history[-1] < fwi.functional_history[0]

    # The solver's own gradient is untouched by the mask; the driver's is
    # the one the optimizers see.
    gradient = fwi.wave.gradient_solve(adjoint_type=adjoint_type)
    assert np.any(gradient.dat.data_ro[frozen] != 0.0)
    gradient = fwi.get_gradient(save=False)
    assert np.all(gradient.dat.data_ro[frozen] == 0.0)
    assert np.any(gradient.dat.data_ro[~frozen] != 0.0)


def test_set_gradient_mask_box(tmp_path, monkeypatch):
    """A box keeps the model inside it and freezes it beyond its sides.

    Without arguments the box is the domain without its absorbing layer,
    which a solver without one cannot provide.
    """
    monkeypatch.chdir(tmp_path)
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)
    space = fwi.wave.c.function_space()
    z, x = fire.SpatialCoordinate(fwi.wave.mesh)

    with pytest.raises(ValueError, match="absorbing layer"):
        fwi.set_gradient_mask()
    with pytest.raises(ValueError, match="not both"):
        fwi.set_gradient_mask(1.0, boundaries={"z_min": -0.5})
    with pytest.raises(ValueError, match="not a boundary"):
        fwi.set_gradient_mask(boundaries={"z_top": 0.0})

    fwi.set_gradient_mask(boundaries={"z_min": -0.75, "x_min": 0.25, "x_max": 0.75})
    (mask,) = fwi._gradient_masks(fwi._controlled_parameters(), [space])
    # The sides themselves are kept: the gradient is zeroed beyond them.
    inside = fire.Function(space).interpolate(
        fire.conditional(fire.And(z >= -0.75, fire.And(x >= 0.25, x <= 0.75)), 1.0, 0.0),
    )
    assert np.array_equal(mask.dat.data_ro, inside.dat.data_ro)
    assert 0.0 < mask.dat.data_ro.mean() < 1.0


@pytest.mark.newer_firedrake
def test_fwi_elastic_automated_adjoint(tmp_path, monkeypatch):
    """Invert three elastic parameters at once, each within its own bounds.

    Density and the two wave speeds are on different scales, so the bounds
    take one entry per control, and each parameter comes back as a control of
    its own.
    """
    vmin = [1.0, 1.5, 0.5]
    vmax = [3.0, 4.0, 2.5]
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(ELASTIC_GUESS),
        wave_class=spyro.IsotropicWave,
    )
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({
        Parameter(name): value for name, value in ELASTIC_REAL.items()
    })
    fwi.generate_real_shot_record(save_shot_record=False)

    # The elastic guess model comes from the input dictionary, not a setter.
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    # One control per parameter the equation is written in, in that order.
    assert isinstance(result, list)
    assert [control.name() for control in result] == list(ELASTIC_GUESS)

    for control, low, high, start in zip(
        result, vmin, vmax, ELASTIC_GUESS.values(),
    ):
        values = control.dat.data_ro
        assert values.min() >= low - 1e-10
        assert values.max() <= high + 1e-10
        assert not np.allclose(values, start), (
            f"{control.name()} was left where it started."
        )

    assert len(fwi.functional_history) > 1
    assert fwi.functional_history[-1] < fwi.functional_history[0]

    # The optimum is written back into the solver's own material parameters.
    for control, parameter in zip(
        result, (fwi.wave.rho, fwi.wave.c, fwi.wave.c_s),
    ):
        assert np.allclose(control.dat.data_ro, parameter.dat.data_ro)


@pytest.mark.newer_firedrake
def test_fwi_elastic_gradient_mask_per_control(tmp_path, monkeypatch):
    """A gradient mask given per control holds the controls it zeroes.

    Zeroing one control's gradient is how an inversion of several
    parameters is run one parameter at a time: the tape is recorded with
    all of them, and each stage moves only the ones its mask lets through.
    """
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(ELASTIC_GUESS),
        wave_class=spyro.IsotropicWave,
    )
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({
        Parameter(name): value for name, value in ELASTIC_REAL.items()
    })
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    settings = dict(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=[1.0, 1.5, 0.5], vmax=[3.0, 4.0, 2.5], maxiter=2,
    )
    # A mask for a parameter that is not a control is a mistake, not a no-op.
    with pytest.raises(ValueError, match="not controls"):
        fwi.run_fwi(gradient_mask={Parameter.LAMBDA: 0.0}, **settings)

    # The p-wave velocity is held; the mask says nothing about the others.
    rho, cp, cs = fwi.run_fwi(
        gradient_mask={Parameter.P_WAVE_VELOCITY: 0.0}, **settings,
    )
    assert np.allclose(cp.dat.data_ro, ELASTIC_GUESS["p_wave_velocity"]), (
        "the p-wave velocity moved although its gradient is masked out"
    )
    for control, start in ((rho, ELASTIC_GUESS["density"]),
                           (cs, ELASTIC_GUESS["s_wave_velocity"])):
        assert not np.allclose(control.dat.data_ro, start), (
            f"{control.name()} did not move although it is not masked"
        )
    assert fwi.functional_history[-1] < fwi.functional_history[0]

    # The next stage moves the p-wave velocity from where the last one
    # left the others, on a new recording of the same tape.
    cs_held = cs.copy(deepcopy=True)
    rho, cp, cs = fwi.run_fwi(
        gradient_mask={Parameter.DENSITY: 0.0, Parameter.S_WAVE_VELOCITY: 0.0},
        **settings,
    )
    assert not np.allclose(cp.dat.data_ro, ELASTIC_GUESS["p_wave_velocity"])
    assert np.allclose(cs.dat.data_ro, cs_held.dat.data_ro)


@pytest.mark.newer_firedrake
def test_fwi_elastic_stages(tmp_path, monkeypatch):
    """One ``run_fwi`` in stages, each moving only some of the controls.

    The tape is recorded once with every control. The first stage moves
    the s-wave velocity alone and the second the p-wave velocity alone,
    from where the first left the s-wave velocity; the density is moved by
    neither. The iteration count and the functional history run through
    both stages as one run.
    """
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(ELASTIC_GUESS),
        wave_class=spyro.IsotropicWave,
    )
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({
        Parameter(name): value for name, value in ELASTIC_REAL.items()
    })
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    bounds = dict(vmin=[1.0, 1.5, 0.5], vmax=[3.0, 4.0, 2.5])
    settings = dict(adjoint_type=AdjointType.AUTOMATED_ADJOINT, **bounds)
    S, P = Parameter.S_WAVE_VELOCITY, Parameter.P_WAVE_VELOCITY

    # Stages are run by TAO, carry their own budgets, and are checked.
    with pytest.raises(ValueError, match="AUTOMATED_ADJOINT"):
        fwi.run_fwi(stages=[(S, 1)], **bounds)
    with pytest.raises(ValueError, match="maxiter"):
        fwi.run_fwi(stages=[(S, 1)], maxiter=1, **settings)
    with pytest.raises(ValueError, match="pair"):
        fwi.run_fwi(stages=[S], **settings)
    with pytest.raises(ValueError, match="not controls"):
        fwi.run_fwi(stages=[(Parameter.LAMBDA, 1)], **settings)
    with pytest.raises(ValueError, match="positive"):
        fwi.run_fwi(stages=[(S, 0)], **settings)

    rho, cp, cs = fwi.run_fwi(stages=[(S, 2), (P, 1)], **settings)

    assert fwi.current_iteration == 3
    assert len(fwi.functional_history) == 4
    assert all(
        later < earlier for earlier, later in zip(
            fwi.functional_history, fwi.functional_history[1:],
        )
    ), "every stage brought the functional down"
    assert np.allclose(rho.dat.data_ro, ELASTIC_GUESS["density"]), (
        "the density moved although no stage moves it"
    )
    for control, start in ((cp, ELASTIC_GUESS["p_wave_velocity"]),
                           (cs, ELASTIC_GUESS["s_wave_velocity"])):
        assert not np.allclose(control.dat.data_ro, start), (
            f"{control.name()} did not move in its stage"
        )

    # A held control that starts outside its bounds is moved onto them by
    # the optimizer, which the stage refuses to pass off as a hold.
    with pytest.raises(RuntimeError, match="hold"):
        fwi.run_fwi(
            stages=[(S, 1)], adjoint_type=AdjointType.AUTOMATED_ADJOINT,
            vmin=[ELASTIC_GUESS["density"] + 0.5, 1.5, 0.5], vmax=[3.0, 4.0, 2.5],
        )


@pytest.mark.newer_firedrake
@pytest.mark.parametrize("checkpointing", [False, True])
def test_next_stage_reads_the_parameter_the_last_one_left(
    tmp_path, monkeypatch, checkpointing,
):
    """A parameter inverted for in one stage keeps its value in the next.

    A two-stage inversion moves one parameter, then another with the first
    held where it got to. Recording the first stage leaves a checkpoint of
    the parameter on its block variable, and under a checkpoint schedule
    that is a copy, which survives the field moving on. The second stage's
    recording has to read the field as it is -- including on replay, which
    is what the optimizer evaluates -- although the field is no longer a
    control.
    """
    monkeypatch.chdir(tmp_path)

    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(ELASTIC_GUESS),
        wave_class=spyro.IsotropicWave,
    )
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({
        Parameter(name): value for name, value in ELASTIC_REAL.items()
    })
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control({
        Parameter.P_WAVE_VELOCITY: ELASTIC_GUESS["p_wave_velocity"],
        Parameter.S_WAVE_VELOCITY: ELASTIC_GUESS["s_wave_velocity"],
    })

    # First stage: the s-wave velocity alone.
    fwi.wave.enable_automated_adjoint(
        control_parameters={Parameter.S_WAVE_VELOCITY},
        checkpointing=checkpointing,
    )
    fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=0.5, vmax=2.5, maxiter=1,
    )
    # The field moves on from whatever the first stage last evaluated.
    fwi.set_guess_control({
        Parameter.S_WAVE_VELOCITY: ELASTIC_REAL["s_wave_velocity"],
    })

    # Second stage: the p-wave velocity, with the s-wave velocity held.
    fwi.wave.enable_automated_adjoint(
        control_parameters={Parameter.P_WAVE_VELOCITY},
        checkpointing=checkpointing,
    )
    fwi.wave.forward_solve()
    taped = float(fwi.wave.functional_value)
    functional = fwi.wave.automated_adjoint.create_reduced_functional(
        fwi.wave.functional_value,
    )
    replayed = float(functional(fwi.wave.c))
    assert np.isclose(replayed, taped, rtol=1e-10), (
        "the replay read the s-wave velocity the first stage left on the "
        "tape, not the field's current value"
    )
