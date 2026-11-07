"""Eikonal source snapping and boundary minimum under spatial parallelism.

Run with ``mpiexec -n 2 pytest tests/parallel/test_eikonal_parallel.py``. The
results must not depend on the number of ranks, so the expected values are
those of a serial run.
"""

import firedrake as fire
import numpy as np
import pytest

from spyro.abc.eik_min import Minimum_Eikonal
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.eikonal.eikonal_eq import Eikonal_Modeling
from spyro.utils.utils import global_min
from tests.on_one_core.test_eikonal_new import wave_dict

pytestmark = pytest.mark.parallel(2)


def square_space() -> fire.FunctionSpace:
    """Build a P1 space on a 4 km x 4 km mesh with nodes at integer coordinates.

    Returns
    -------
    `Firedrake.FunctionSpace`
        P1 space on a 4 x 4 cell mesh, so that distances between nodes and
        Eikonal ties are exact in floating point.
    """
    return fire.FunctionSpace(fire.RectangleMesh(4, 4, 4.0, 4.0), "CG", 1)


def marked_points(marker: fire.Function) -> list[list[float]]:
    """Return the coordinates of the nodes marked as sources on all ranks.

    Parameters
    ----------
    marker : `Firedrake.Function`
        Source marker, 1 at the source nodes and 0 elsewhere.

    Returns
    -------
    `list`
        Sorted coordinates of the marked nodes, identical on every rank.
    """
    V = marker.function_space()
    mesh = V.mesh()
    coords = fire.assemble(
        fire.interpolate(
            fire.SpatialCoordinate(mesh),
            fire.VectorFunctionSpace(mesh, V.ufl_element()),
        )
    )
    owned = np.flatnonzero(marker.dat.data_ro)
    points = coords.dat.data_ro[owned].tolist()
    return sorted(sum(mesh.comm.allgather(points), []))


@pytest.mark.parametrize(
    "source, expected",
    [
        ((2.0, 1.0), [[2.0, 1.0]]),  # on a node
        ((2.2, 0.9), [[2.0, 1.0]]),  # off node, unique nearest
        ((2.5, 1.0), [[2.0, 1.0], [3.0, 1.0]]),  # equidistant to two nodes
    ],
)
def test_eikonal_bcs_snaps_source_to_nearest_node(
    source: tuple[float, float], expected: list[list[float]]
) -> None:
    """Check that each source is fixed at its nearest node(s), with halos.

    Parameters
    ----------
    source : `tuple`
        Source coordinates.
    expected : `list`
        Coordinates of the nodes that must hold the Dirichlet condition.

    Returns
    -------
    None
    """
    V = square_space()
    bcs, marker = Eikonal_Modeling(2, [source], degree_eik=1).eikonal_bcs(V, lmin=1.0)

    assert marked_points(marker) == expected

    # The BC must fix the same nodes the marker shows, halo copies included
    fixed = fire.Function(V).assign(1.0)
    for bc in bcs:
        bc.apply(fixed)
    np.testing.assert_array_equal(
        fixed.dat.data_ro_with_halos, 1.0 - marker.dat.data_ro_with_halos
    )


def test_eikonal_bcs_rejects_source_far_from_mesh() -> None:
    """Check that a source with no node within ``lmin / (degree_eik + 1)`` raises.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="no mesh node"):
        Eikonal_Modeling(2, [(9.0, 9.0)], degree_eik=1).eikonal_bcs(
            square_space(), lmin=1.0
        )


@pytest.mark.parametrize(
    "family, eikonal, eikmin, pnt_crit",
    [
        ("CG", lambda x: (x - 1.3) ** 2, 0.09, [0.0, 1.0]),  # unique minimum
        ("DG", lambda x: fire.Constant(1.0), 1.0, [0.0, 0.0]),  # every node ties
    ],
)
def test_ident_eik_on_bnd(
    family: str, eikonal, eikmin: float, pnt_crit: list[float]
) -> None:
    """Check the boundary minimum and its partition-independent tie-break.

    On the boundary ``z = 0`` (nodes at ``x = 0, ..., 4``) the minimum is
    reduced over all ranks. When several nodes tie, the critical point is the
    lexicographically smallest one. A constant DG0 field makes every node tie
    exactly, since point evaluation of a P1 field at a vertex is not exact.

    Parameters
    ----------
    family : `str`
        Family of the Eikonal field, "CG" (degree 1) or "DG" (degree 0).
    eikonal : `callable`
        Eikonal field as a function of the coordinate ``x``.
    eikmin : `float`
        Expected minimum Eikonal on the boundary.
    pnt_crit : `list`
        Expected critical point.

    Returns
    -------
    None
    """
    V = square_space()
    mesh = V.mesh()
    z, x = fire.SpatialCoordinate(mesh)

    # Only the attributes ident_eik_on_bnd reads; building a full wave is not
    # needed to exercise the reduction
    eik = Minimum_Eikonal.__new__(Minimum_Eikonal)
    eik.dimension = 2
    eik.mesh = mesh
    eik.node_tol = 1e-6
    eik.yp = fire.Function(
        fire.FunctionSpace(mesh, family, 1 if family == "CG" else 0)
    ).interpolate(eikonal(x))
    eik.c = fire.Function(V).interpolate(1.0 + x)
    eik.node_positions = fire.assemble(
        fire.interpolate(
            fire.SpatialCoordinate(mesh), fire.VectorFunctionSpace(mesh, "CG", 1)
        )
    ).dat.data_ro_with_halos
    bnd_ids = fire.DirichletBC(V, 0.0, 1).nodes

    result = eik.ident_eik_on_bnd(bnd_ids)

    assert result[0] == pytest.approx(eikmin, rel=1e-12)
    assert result[1].tolist() == pnt_crit
    assert result[2] == pytest.approx(1.0 + pnt_crit[1], rel=1e-12)
    result = (result[0], result[1].tolist(), result[2])
    assert mesh.comm.allgather(result) == [result] * mesh.comm.size


def test_global_min_reduces_over_ranks() -> None:
    """Check that the minimum covers the nodes owned by every rank.

    Returns
    -------
    None
    """
    V = square_space()
    z, x = fire.SpatialCoordinate(V.mesh())
    assert global_min(fire.Function(V).interpolate(10.0 - z - x)) == 2.0


def critical_boundary_data(geometry: str, degree: int, f_est: float) -> list:
    """Run a 2D Eikonal analysis and return its boundary data.

    The setup is that of ``test_eikonal_new`` with a smooth velocity
    interpolated straight into the continuous space. Evaluating a discontinuous
    or DG0 velocity at nodes shared by several cells depends on the partition,
    which would hide the behaviour under test.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".
    degree : `int`
        Finite element order for the Eikonal equation.
    f_est : `float`
        Factor for the stabilizing term in the Eikonal equation.

    Returns
    -------
    `list`
        ``[pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]`` per boundary,
        sorted by the minimum Eikonal.
    """
    dictionary = wave_dict(geometry, 2, degree, "consistent")
    dictionary["visualization"]["output_folder"] += f"_np{fire.COMM_WORLD.size}"
    wave = AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(
        conditional=3.0 - 1.5 * wave.mesh_x, dg_velocity_model=False
    )
    wave.mesh_ops.preamble_mesh_operations(wave, ele_type_eik="consistent", f_est=f_est)
    wave.layer_ops.critical_boundary_points(wave)
    return wave.layer_ops.eik_bnd


# Serial results of critical_boundary_data:
# (critical point, minimum Eikonal in s) per absorbing boundary
@pytest.mark.parametrize(
    "geometry, degree, f_est, serial",
    [
        (
            "T",
            2,
            0.06,
            [
                ([-0.5, 0.0], 0.08827854817158261),
                ([-1.0, 0.15000000000000002], 0.19146894159595299),
                ([-0.45, 1.0], 0.37796435804532014),
            ],
        ),
        (
            "Q",
            2,
            0.05,
            [
                ([-0.5, 0.0], 0.08929751297228099),
                ([-1.0, 0.15000000000000002], 0.1928097578035655),
                ([-0.5, 1.0], 0.37804633159604933),
            ],
        ),
    ],
)
def test_eikonal_critical_points_match_serial(
    geometry: str, degree: int, f_est: float, serial: list
) -> None:
    """Check that the parallel Eikonal analysis reproduces the serial result.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".
    degree : `int`
        Finite element order for the Eikonal equation.
    f_est : `float`
        Factor for the stabilizing term in the Eikonal equation.
    serial : `list`
        Serial ``(pnt_crit, eikmin)`` per boundary, sorted by ``eikmin``.

    Returns
    -------
    None
    """
    eik_bnd = critical_boundary_data(geometry, degree, f_est)

    result = [(list(pnt), eikmin) for pnt, _, eikmin, *_ in eik_bnd]
    assert [pnt for pnt, _ in result] == [pnt for pnt, _ in serial]
    np.testing.assert_allclose(
        [e for _, e in result], [e for _, e in serial], rtol=1e-12
    )
    assert fire.COMM_WORLD.allgather(result) == [result] * fire.COMM_WORLD.size
