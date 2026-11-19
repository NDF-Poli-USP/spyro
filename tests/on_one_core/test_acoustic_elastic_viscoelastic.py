from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.solvers.elastic_wave.tensor_computation import (
    C_tti_tensor,
    C_vti_tensor,
    Gamma_isotropic,
    Gamma_TTI,
    Gamma_VTI,
)

from .test_acoustic_elastic_tti import ANISOTROPY, _dictionary, _relative_difference

ISOTROPIC = {"epsilon": None, "delta": None, "gamma": None}
MEDIA = {"iso": (None, ISOTROPIC), "vti": (None, {}), "tti": (30.0, {})}


def _gsls_fit(
    fmin: float, fmax: float, branches: int, samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit GSLS weights for a constant quality factor in a frequency band.

    Parameters
    ----------
    fmin : float
        Lowest frequency of the band, in Hz.
    fmax : float
        Highest frequency of the band, in Hz.
    branches : int
        Number of relaxation mechanisms.
    samples : int, optional
        Number of frequencies in the least-squares fit.

    Returns
    -------
    numpy.ndarray
        Weights ``y_l`` of the mechanisms.
    numpy.ndarray
        Relaxation angular frequencies ``omega_l``.
    """
    omega_relax = 2 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax), branches)
    omega = 2 * np.pi * np.linspace(fmin, fmax, samples)
    A = omega_relax * omega[:, None] / (omega[:, None] ** 2 + omega_relax**2)
    y = np.linalg.lstsq(A, np.ones(samples), rcond=None)[0]
    return y, omega_relax


def _viscoelastic_dictionary(
    medium: str, inverse_Q: float | None, final_time: float = 0.25
) -> dict:
    """Build a coupled dictionary with an elastic or viscoelastic solid.

    Parameters
    ----------
    medium : str
        Solid medium: "iso", "vti" or "tti".
    inverse_Q : float or None
        Reciprocal quality factor of every solid parameter. ``None`` gives
        an elastic solid.
    final_time : float, optional
        Final simulation time.

    Returns
    -------
    dict
        Spyro input dictionary.
    """
    theta, overrides = MEDIA[medium]
    dictionary = _dictionary(theta, **overrides)
    dictionary["time_axis"]["final_time"] = final_time
    if inverse_Q is not None:
        y, omega = _gsls_fit(2.0, 30.0, 3)
        dictionary["viscoelastic"] = True
        dictionary["synthetic_data"].update(
            {
                name: inverse_Q
                for name in ("Q_vp", "Q_vs", "Q_epsilon", "Q_delta", "Q_gamma")
            }
        )
        dictionary["viscoelasticity"] = {
            "visco_type": "maxwell_gsls_Q",
            "branches": len(y),
            "omega_gsls": list(omega),
            "y_gsls": list(y),
        }
    return dictionary


def _solid_record(
    medium: str,
    inverse_Q: float | None,
    use_monolithic: bool = False,
    final_time: float = 0.25,
) -> np.ndarray:
    """Run a coupled forward solve and return the solid receiver history.

    Parameters
    ----------
    medium : str
        Solid medium: "iso", "vti" or "tti".
    inverse_Q : float or None
        Reciprocal quality factor of every solid parameter. ``None`` gives
        an elastic solid.
    use_monolithic : bool, optional
        Whether to use the monolithic scheme instead of the sequential one.
    final_time : float, optional
        Final simulation time.

    Returns
    -------
    numpy.ndarray
        Solid displacement at the receivers for every time step.
    """
    wave = AcousticElasticWave(_viscoelastic_dictionary(medium, inverse_Q, final_time))
    wave.use_monolithic = use_monolithic
    wave.forward_solve()
    return np.asarray(wave.solid_receiver_history)


def _gamma(theta: float | None, **thomsen) -> np.ndarray:
    """Evaluate the 2D GSLS factors of a VTI (or TTI) solid.

    The Q of the vertical P wave differs from the others, so a factor in
    the wrong Voigt position is detected.

    Parameters
    ----------
    theta : float or None
        Tilt angle in degrees. ``None`` gives a VTI solid.
    **thomsen
        Entries replacing the default Thomsen parameters.

    Returns
    -------
    numpy.ndarray
        The 3x3 GSLS factors.
    """
    material = SimpleNamespace(
        rho=fire.Constant(2.0),
        c=fire.Constant(2.0),
        c_s=fire.Constant(1.2),
        theta=theta,
        phi=None,
        Q_vp=fire.Constant(0.01),
        Q_vs=fire.Constant(0.02),
        Q_epsilon=fire.Constant(0.03),
        Q_delta=fire.Constant(0.04),
        Q_gamma=fire.Constant(0.05),
        **{**ANISOTROPY, **thomsen},
    )
    C_vti = C_vti_tensor(material, 2)
    if theta is None:
        Gamma = Gamma_VTI(material, C_vti, 2)
    else:
        Gamma = Gamma_TTI(material, 2, C_tti_tensor(material, C_vti, 2))
    mesh = fire.UnitSquareMesh(1, 1)
    space = fire.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3))
    return fire.Function(space).interpolate(Gamma).dat.data_ro[0].copy()


def test_vti_gamma_follows_stiffness_ordering():
    """The VTI factors must sit in the Voigt order (zz, xx, zx) of C_vti."""
    Gamma = _gamma(None)
    epsilon = ANISOTROPY["epsilon"]

    assert np.isclose(Gamma[0, 0], 0.01)  # vertical P: Q_vp
    assert np.isclose(Gamma[1, 1], 2 * epsilon / (1 + 2 * epsilon) * 0.03 + 0.01)
    assert np.isclose(Gamma[2, 2], 0.02)  # vertical S: Q_vs


@pytest.mark.parametrize("anisotropy_type", ["exact", "weak"])
def test_vti_gamma_isotropic_limit(anisotropy_type):
    """Zero Thomsen parameters must give the isotropic GSLS factors."""
    Gamma_vti = _gamma(
        None, epsilon=0.0, delta=0.0, gamma=0.0, anisotropy_type=anisotropy_type
    )
    material = SimpleNamespace(
        c=fire.Constant(2.0),
        c_s=fire.Constant(1.2),
        Q_vp=fire.Constant(0.01),
        Q_vs=fire.Constant(0.02),
    )
    mesh = fire.UnitSquareMesh(1, 1)
    space = fire.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3))
    Gamma_iso = (
        fire.Function(space).interpolate(Gamma_isotropic(material, 2)).dat.data_ro[0]
    )

    assert np.allclose(Gamma_vti, Gamma_iso)


def test_tti_gamma_rotates_vti_gamma():
    """A zero tilt keeps the VTI factors and 90 degrees swaps zz and xx."""
    Gamma_vti = _gamma(None)
    Gamma_tilt_0 = _gamma(0.0)
    Gamma_tilt_90 = _gamma(90.0)

    assert np.allclose(Gamma_tilt_0, Gamma_vti)
    assert np.isclose(Gamma_tilt_90[0, 0], Gamma_vti[1, 1])
    assert np.isclose(Gamma_tilt_90[1, 1], Gamma_vti[0, 0])
    assert np.isclose(Gamma_tilt_90[0, 1], Gamma_vti[0, 1])
    assert np.isclose(Gamma_tilt_90[2, 2], Gamma_vti[2, 2])


@pytest.mark.parametrize("medium", ["iso", "vti", "tti"])
def test_infinite_Q_matches_elastic(medium):
    """With zero reciprocal Q the viscoelastic solid must be the elastic one."""
    elastic = _solid_record(medium, None)
    viscoelastic = _solid_record(medium, 0.0)

    assert np.linalg.norm(elastic) > 0.0
    assert _relative_difference(viscoelastic, elastic) < 1e-10


@pytest.mark.parametrize("medium", ["iso", "tti"])
def test_monolithic_matches_sequential_for_viscoelastic_solid(medium):
    """Monolithic and sequential schemes must agree for a viscoelastic solid."""
    sequential = _solid_record(medium, 0.05)
    monolithic = _solid_record(medium, 0.05, use_monolithic=True)

    assert _relative_difference(monolithic, sequential) < 1e-8


@pytest.mark.parametrize("medium", ["vti", "tti"])
def test_lower_Q_attenuates_more(medium):
    """The late solid reverberation must decrease as Q decreases."""
    final_time = 1.0
    records = [
        _solid_record(medium, inverse_Q, final_time=final_time)
        for inverse_Q in (None, 1 / 50, 1 / 10)
    ]
    half = len(records[0]) // 2
    late = [np.linalg.norm(r[half:]) for r in records]

    assert late[0] > late[1] > late[2]
    assert late[2] < 0.9 * late[0]


def test_memory_variables_live_on_solid_submesh():
    """The GSLS memory must be built on the solid submesh, one per branch."""
    wave = AcousticElasticWave(_viscoelastic_dictionary("vti", 0.05))
    wave._initialize_model_parameters()
    wave.matrix_building()

    assert len(wave.zeta_list) == 3
    for zeta in wave.zeta_list:
        assert zeta.function_space().mesh() is wave.submesh_solid
    assert wave.eps_n.function_space() == wave.strain_space


def test_inconsistent_branches_raise():
    """A branch count that does not match the GSLS lists must be rejected."""
    dictionary = _viscoelastic_dictionary("iso", 0.05)
    dictionary["viscoelasticity"]["branches"] = 2
    wave = AcousticElasticWave(dictionary)
    wave._initialize_model_parameters()

    with pytest.raises(ValueError, match="per branch"):
        wave.matrix_building()
