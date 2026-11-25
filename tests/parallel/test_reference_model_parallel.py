"""Reference model of the ABCs under spatial parallelism.

Run with ``mpiexec -n 2 pytest tests/parallel/test_reference_model_parallel.py``.
The reference ("infinite") model must not depend on the number of ranks, so
the expected values are those of the serial ``test_reference_model_abc``.
"""

import firedrake as fire
import numpy as np
import pytest
from mpi4py import MPI

from spyro.utils.utils import global_max, global_min
from spyro.version_control import is_firedrake_new
from tests.on_one_core import test_reference_model_abc as trm
from tests.on_one_core.test_velocity_profile import (
    check_velocity_profile,
    velocity_profile,
)

pytestmark = pytest.mark.parallel(2)

# Serial minimum Eikonal in s of the 2D cases of test_reference_model_abc. For T,
# it differs between Firedrake versions (see the version notes there)
SERIAL_EIKMIN = {"T": 0.084544 if is_firedrake_new() else 0.084586, "Q": 0.093368}


@pytest.fixture
def parallel_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the outputs of the parallel runs apart from the serial ones.

    Parameters
    ----------
    monkeypatch : `pytest.MonkeyPatch`
        Fixture that replaces `wave_dict` in ``test_reference_model_abc``.

    Returns
    -------
    None
    """
    original = trm.wave_dict
    suffix = f"_np{fire.COMM_WORLD.size}"

    def wave_dict_parallel(*args) -> dict:
        """Return the serial dictionary with the output paths suffixed.

        Parameters
        ----------
        *args
            Arguments of `wave_dict` in ``test_reference_model_abc``.

        Returns
        -------
        `dict`
            Dictionary of the serial test, with ``_np<ranks>`` after the
            output folder.
        """
        dictionary = original(*args)
        visualization = dictionary["visualization"]
        folder = visualization["output_folder"]
        visualization["output_folder"] = folder + suffix
        visualization["acoustic_energy_filename"] = visualization[
            "acoustic_energy_filename"
        ].replace(folder, folder + suffix, 1)
        return dictionary

    monkeypatch.setattr(trm, "wave_dict", wave_dict_parallel)


def global_coordinate_range(mesh: fire.MeshGeometry) -> list[tuple[float, float]]:
    """Return the range of each coordinate of a mesh over every rank.

    Parameters
    ----------
    mesh : `Firedrake.MeshGeometry`
        Mesh distributed over the ranks.

    Returns
    -------
    `list`
        ``(minimum, maximum)`` of each coordinate, identical on every rank.
    """
    coords = mesh.coordinates.dat.data_ro
    return [
        (
            mesh.comm.allreduce(coords[:, i].min(initial=np.inf), op=MPI.MIN),
            mesh.comm.allreduce(coords[:, i].max(initial=-np.inf), op=MPI.MAX),
        )
        for i in range(coords.shape[1])
    ]


@pytest.mark.parametrize("calc_eik", [True, False])
@pytest.mark.parametrize("element_geometry", ["T", "Q"])
def test_reference_model_agrees_on_all_ranks(
    parallel_output: None, element_geometry: str, calc_eik: bool
) -> None:
    """Check that every stage of the reference model gives the serial values.

    The stages of ``infinite_model`` before the forward problem are run on the
    2D case of ``test_reference_model_abc``, with the velocity 3.0 km/s for
    x < 0.5 and 1.5 km/s otherwise. Each one reduces data that every rank only
    holds in part: the velocity ranges of the domain and of its boundary, the
    minimum Eikonal, the stable time step, the size of the extension, the
    extended mesh and the velocity extended into it. Every rank must find the
    serial values.

    Parameters
    ----------
    parallel_output : None
        Fixture that writes the outputs apart from the serial ones.
    element_geometry : `str`
        Geometry of the finite element, "T" or "Q".
    calc_eik : `bool`
        If `True`, the extension is based on the Eikonal analysis; otherwise,
        on the distance from the source to the boundary and ``c_max``.

    Returns
    -------
    None

    Notes
    -----
    The velocity of the extension comes from the original boundary, so its
    range must stay within the boundary range, 1.5 to 3.0 km/s.
    """
    case = (2, element_geometry, "ARNOLDI")
    wave = trm.wave_instance(element_geometry, 2, "hybrid", calc_eik, "ARNOLDI")
    layer = wave.layer_ops

    # Stages of infinite_model before the forward problem
    layer.check_timestep_abc(wave, max_divisor_tf=1, method="ARNOLDI")
    layer.geometry_infinite_model(wave)
    layer.create_mesh_with_layer(wave, inf_model=True)
    layer.velocity_abc(wave, inf_model=True)

    result = {
        "ranges": (wave.c_min, wave.c_max, wave.c_bnd_min, wave.c_bnd_max),
        "eikmin": layer.eik_bnd[0][2] if calc_eik else None,
        "dt": wave.dt,
        "pad": layer.abc_pad_length,
        "mesh": global_coordinate_range(wave.mesh),
        "c_extended": (global_min(wave.c), global_max(wave.c)),
    }

    assert result["ranges"] == (1.5, 3.0, 1.5, 3.0)
    if calc_eik:
        np.testing.assert_allclose(
            result["eikmin"], SERIAL_EIKMIN[element_geometry], rtol=1e-4
        )
    assert np.isclose(
        result["dt"], trm.case_parameter(trm.REFERENCE_CONFIG["dt"], case)
    )
    expected_pad = trm.case_parameter(trm.REFERENCE_CONFIG["pad_length"], case)
    assert np.isclose(result["pad"], expected_pad)
    np.testing.assert_allclose(
        result["mesh"],
        [(-1.0 - expected_pad, 0.0), (-expected_pad, 1.0 + expected_pad)],
        atol=1e-12,
    )
    np.testing.assert_allclose(result["c_extended"], (1.5, 3.0), rtol=1e-12)
    assert fire.COMM_WORLD.allgather(result) == [result] * fire.COMM_WORLD.size


@pytest.mark.parametrize("element_geometry", ["T", "Q"])
def test_velocity_profile_agrees_on_all_ranks(
    parallel_output: None, element_geometry: str
) -> None:
    """Check that the velocity of the reference model is that of a serial run.

    The DG0 velocity is converted to the continuous space in the preamble and
    after the extension of the infinite model. Every node shared by cells of
    different velocities must take the minimum of them on every rank, and the
    integrals of the velocity must be the serial ones (see
    ``tests/on_one_core/test_velocity_profile.py``).

    Parameters
    ----------
    parallel_output : None
        Fixture that writes the outputs apart from the serial ones.
    element_geometry : `str`
        Geometry of the finite element, "T" or "Q".

    Returns
    -------
    None
    """
    profile = velocity_profile(element_geometry)

    check_velocity_profile(profile, element_geometry)
    assert fire.COMM_WORLD.allgather(profile) == [profile] * fire.COMM_WORLD.size


@pytest.mark.parametrize("calc_eik", [True, False])
def test_infinite_model_abc_in_parallel(
    request: pytest.FixtureRequest, parallel_output: None, calc_eik: bool
) -> None:
    """Run the serial checks of the reference model on two ranks.

    ``test_infinite_model_abc`` builds and solves the reference model with the
    hybrid and the PML schemes, and checks the selected time step, the size of
    the extension and that both schemes give the same receiver signals and
    energies.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Fixture passed on to ``test_infinite_model_abc``.
    parallel_output : None
        Fixture that writes the outputs apart from the serial ones.
    calc_eik : `bool`
        If `True`, the extension is based on the Eikonal analysis.

    Returns
    -------
    None
    """
    trm.test_infinite_model_abc(request, "T", 2, calc_eik, "ARNOLDI")
