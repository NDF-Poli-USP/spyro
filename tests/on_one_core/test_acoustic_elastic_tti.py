from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.solvers.elastic_wave.tensor_computation import C_tti_tensor, C_vti_tensor
from spyro.utils.typing import WaveType

ANISOTROPY = {"epsilon": 0.3, "delta": 0.15, "gamma": 0.1, "anisotropy_type": "exact"}


def _dictionary(theta: float | None, **synthetic_data_overrides) -> dict:
    """Build a small 2D acoustic-elastic dictionary with a VTI/TTI solid.

    Parameters
    ----------
    theta : float or None
        Tilt angle in degrees. ``None`` gives a VTI solid.
    **synthetic_data_overrides
        Entries that replace or extend the default ``synthetic_data``.

    Returns
    -------
    dict
        Spyro input dictionary.
    """
    synthetic_data = {
        "type": "object",
        "bulk_modulus": 2.25,
        "density_fluid": 1.0,
        "density_solid": 2.0,
        "p_wave_velocity": 2.0,
        "s_wave_velocity": 1.2,
        **ANISOTROPY,
    }
    if theta is not None:
        synthetic_data["theta"] = theta
    synthetic_data.update(synthetic_data_overrides)
    return {
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
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
            "edge_length": 0.05,
            "interface_z": -0.5,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.4, 0.5)],
            "frequency": 10.0,
            "delay": 0.1,
            "delay_type": "time",
            "amplitude": 1.0,
            "receiver_locations": [(-0.3, 0.5)],
            "solid_receiver_locations": [(-0.6, 0.5)],
            "use_vertex_only_mesh": False,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.25,
            "dt": 0.001,
            "output_frequency": 1000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "debug_output": False,
        },
        "synthetic_data": synthetic_data,
    }


def _stiffness(theta: float | None) -> tuple[np.ndarray, WaveType]:
    """Assemble the solid stiffness tensor of a coupled wave object.

    Parameters
    ----------
    theta : float or None
        Tilt angle in degrees. ``None`` gives a VTI solid.

    Returns
    -------
    numpy.ndarray
        The 3x3 Voigt stiffness tensor.
    WaveType
        The wave type selected by ``matrix_building``.
    """
    wave = AcousticElasticWave(_dictionary(theta))
    wave._initialize_model_parameters()
    wave.matrix_building()
    space = fire.TensorFunctionSpace(wave.submesh_solid, "DG", 0, shape=(3, 3))
    C = fire.Function(space).interpolate(wave.Elastic_C)
    return C.dat.data_ro[0].copy(), wave.wave_type


def _forward(
    theta: float | None, use_monolithic: bool = False, **synthetic_data_overrides
) -> AcousticElasticWave:
    """Run a short coupled forward solve.

    Parameters
    ----------
    theta : float or None
        Tilt angle in degrees. ``None`` gives a VTI solid.
    use_monolithic : bool, optional
        Whether to use the monolithic scheme instead of the sequential one.
    **synthetic_data_overrides
        Entries that replace or extend the default ``synthetic_data``.

    Returns
    -------
    AcousticElasticWave
        The wave object after the forward solve.
    """
    wave = AcousticElasticWave(_dictionary(theta, **synthetic_data_overrides))
    wave.use_monolithic = use_monolithic
    wave.forward_solve()
    return wave


def _solid_receivers(theta: float | None, use_monolithic: bool) -> np.ndarray:
    """Run a short forward solve and return the solid receiver history.

    Parameters
    ----------
    theta : float or None
        Tilt angle in degrees. ``None`` gives a VTI solid.
    use_monolithic : bool
        Whether to use the monolithic scheme instead of the sequential one.

    Returns
    -------
    numpy.ndarray
        Solid displacement at the receivers for every time step.
    """
    return np.asarray(_forward(theta, use_monolithic).solid_receiver_history)


def _relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the relative l2 difference between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Array being compared.
    b : numpy.ndarray
        Reference array, assumed nonzero.

    Returns
    -------
    float
        ``||a - b|| / ||b||``.
    """
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _stiffness_3d(theta: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the 3D VTI and TTI stiffness tensors.

    Parameters
    ----------
    theta : float
        Tilt angle in degrees.
    phi : float
        Azimuth angle in degrees.

    Returns
    -------
    numpy.ndarray
        The 6x6 Voigt VTI stiffness tensor.
    numpy.ndarray
        The 6x6 Voigt TTI stiffness tensor.
    """
    rho = 2.0
    material = SimpleNamespace(
        rho=fire.Constant(rho),
        c=fire.Constant(2.0),
        c_s=fire.Constant(1.2),
        theta=theta,
        phi=phi,
        **ANISOTROPY,
    )
    C_vti = C_vti_tensor(material, 3)
    C_tti = C_tti_tensor(material, C_vti, 3)
    mesh = fire.UnitSquareMesh(1, 1)
    space = fire.TensorFunctionSpace(mesh, "DG", 0, shape=(6, 6))
    values = [
        fire.Function(space).interpolate(C).dat.data_ro[0].copy()
        for C in (C_vti, C_tti)
    ]
    return values[0], values[1]


def _stiffness_2d_from_3d_material(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the 2D VTI and TTI tensors for the material of the 3D tests.

    Parameters
    ----------
    theta : float
        Tilt angle in degrees.

    Returns
    -------
    numpy.ndarray
        The 3x3 Voigt VTI stiffness tensor.
    numpy.ndarray
        The 3x3 Voigt TTI stiffness tensor.
    """
    material = SimpleNamespace(
        rho=fire.Constant(2.0),
        c=fire.Constant(2.0),
        c_s=fire.Constant(1.2),
        theta=theta,
        phi=None,
        **ANISOTROPY,
    )
    C_vti = C_vti_tensor(material, 2)
    C_tti = C_tti_tensor(material, C_vti, 2)
    mesh = fire.UnitSquareMesh(1, 1)
    space = fire.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3))
    values = [
        fire.Function(space).interpolate(C).dat.data_ro[0].copy()
        for C in (C_vti, C_tti)
    ]
    return values[0], values[1]


def test_tti_theta_zero_matches_vti():
    """A zero tilt angle must reproduce the VTI stiffness tensor."""
    C_vti, vti_type = _stiffness(None)
    C_tti, tti_type = _stiffness(0.0)

    assert vti_type == WaveType.ANISOTROPIC_VTI_ELASTIC
    assert tti_type == WaveType.ANISOTROPIC_TTI_ELASTIC
    assert np.allclose(C_tti, C_vti)


def test_tti_theta_ninety_swaps_vertical_and_horizontal():
    """A 90 degree tilt must exchange the vertical and horizontal moduli."""
    C_vti, _ = _stiffness(None)
    C_tti, _ = _stiffness(90.0)

    assert np.isclose(C_tti[0, 0], C_vti[1, 1])
    assert np.isclose(C_tti[1, 1], C_vti[0, 0])
    assert np.isclose(C_tti[0, 1], C_vti[0, 1])
    assert np.isclose(C_tti[2, 2], C_vti[2, 2])
    assert np.allclose(C_tti[:2, 2], 0.0, atol=1e-12 * C_vti[0, 0])


@pytest.mark.parametrize("phi", [0.0, 30.0, 90.0])
def test_tti_3d_without_tilt_matches_vti(phi):
    """In 3D the azimuth alone must not change a VTI medium."""
    C_vti, C_tti = _stiffness_3d(0.0, phi)

    assert np.allclose(C_tti, C_vti, rtol=1e-10, atol=1e-12 * C_vti.max())


@pytest.mark.parametrize("phi, plane", [(0.0, [0, 1, 5]), (90.0, [0, 2, 4])])
def test_tti_3d_reduces_to_2d(phi, plane):
    """A 3D tilt in the z-x (phi=0) or z-y (phi=90) plane must match 2D TTI.

    The 3D Voigt order is (zz, xx, yy, xy, zy, zx), so ``plane`` selects the
    (zz, xx, zx) or (zz, yy, zy) entries.
    """
    _, C_tti_2d = _stiffness_2d_from_3d_material(30.0)
    _, C_tti_3d = _stiffness_3d(30.0, phi)

    block = C_tti_3d[np.ix_(plane, plane)]
    assert np.allclose(block, C_tti_2d, rtol=1e-10, atol=1e-12 * C_tti_2d.max())


@pytest.mark.parametrize("theta", [None, 30.0])
def test_monolithic_matches_sequential_for_anisotropic_solid(theta):
    """Monolithic and sequential schemes must agree for VTI and TTI solids."""
    sequential = _solid_receivers(theta, use_monolithic=False)
    monolithic = _solid_receivers(theta, use_monolithic=True)

    assert np.linalg.norm(sequential) > 0.0
    assert _relative_difference(monolithic, sequential) < 1e-8


def test_tti_changes_solid_response():
    """A tilted solid must give a different response than the VTI one."""
    vti = _solid_receivers(None, use_monolithic=False)
    tti = _solid_receivers(30.0, use_monolithic=False)

    assert _relative_difference(tti, vti) > 1e-2


def test_anisotropic_solid_keeps_fluid_velocity():
    """``velocity_fluid`` must survive the anisotropic tensor setup.

    With unit fluid density, ``velocity_fluid = sqrt(bulk_modulus)`` describes
    the same fluid as ``bulk_modulus``/``density_fluid``.
    """
    from_modulus = _forward(30.0)
    from_velocity = _forward(
        30.0, velocity_fluid=1.5, bulk_modulus=None, density_fluid=None
    )

    assert float(from_velocity.c) == 1.5
    fluid = (
        np.asarray(from_velocity.forward_solution_receivers),
        np.asarray(from_modulus.forward_solution_receivers),
    )
    solid = (
        np.asarray(from_velocity.solid_receiver_history),
        np.asarray(from_modulus.solid_receiver_history),
    )
    assert _relative_difference(*fluid) < 1e-10
    assert _relative_difference(*solid) < 1e-10


@pytest.mark.parametrize("theta", [None, 30.0])
def test_derived_fields_match_isotropic_limit(theta):
    """Zero Thomsen parameters must give the isotropic sigma_zz and p_eq."""
    wave = _forward(theta, epsilon=0.0, delta=0.0, gamma=0.0)
    sigma_zz = wave._compute_sigma_zz().dat.data_ro.copy()
    p_equivalent = wave._compute_p_equivalent().dat.data_ro.copy()

    wave.Elastic_C = None
    sigma_zz_isotropic = wave._compute_sigma_zz().dat.data_ro.copy()
    p_equivalent_isotropic = wave._compute_p_equivalent().dat.data_ro.copy()

    assert np.linalg.norm(sigma_zz_isotropic) > 0.0
    assert _relative_difference(sigma_zz, sigma_zz_isotropic) < 1e-10
    assert _relative_difference(p_equivalent, p_equivalent_isotropic) < 1e-10


def test_derived_fields_use_anisotropic_tensor():
    """The TTI sigma_zz must differ from the isotropic formula."""
    wave = _forward(30.0)
    sigma_zz = wave._compute_sigma_zz().dat.data_ro.copy()

    wave.Elastic_C = None
    sigma_zz_isotropic = wave._compute_sigma_zz().dat.data_ro.copy()

    assert _relative_difference(sigma_zz, sigma_zz_isotropic) > 1e-2


def test_coupled_fields_are_written(tmp_path):
    """Requested displacement, p_equivalent and sigma_zz files must be written."""
    fields = ("displacement", "p_equivalent", "sigma_zz")
    dictionary = _dictionary(30.0)
    dictionary["time_axis"]["final_time"] = 0.01
    dictionary["time_axis"]["output_frequency"] = 5
    for key in fields:
        dictionary["visualization"][f"{key}_output"] = True
        dictionary["visualization"][f"{key}_output_filename"] = str(
            tmp_path / f"{key}.pvd"
        )

    AcousticElasticWave(dictionary).forward_solve()

    for key in fields:
        assert list(tmp_path.glob(f"{key}sn*.pvd")), f"{key} output not written"
