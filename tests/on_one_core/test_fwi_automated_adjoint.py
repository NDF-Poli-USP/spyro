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
    """Invert an acoustic velocity model from data a faster one generated.

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
