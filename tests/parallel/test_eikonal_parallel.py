"""Eikonal source conditions and boundary minimum under spatial parallelism.

Run with ``mpiexec -n 2 pytest tests/parallel/test_eikonal_parallel.py``. The
results must not depend on the number of ranks, so the expected values are
those of a serial run or, for the 'local_time' pipeline tests, the
theoretical travel time. Values recorded with one Firedrake installation can
differ slightly with another; the tests that compare recorded values state
the installations and the tolerance in their Notes.
"""

import firedrake as fire
import numpy as np
import pytest
from pyop2.configuration import configuration

from spyro.abc.eik_min import Minimum_Eikonal
from spyro.meshing.meshing_operations import MeshOps
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.eikonal.eikonal_eq import Eikonal_Modeling
from spyro.utils.utils import global_min
from tests.on_one_core.test_eikonal_new import wave_dict

pytestmark = [pytest.mark.parallel(2), pytest.mark.eikonal]


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
    """Check that a source with no node within one cell diameter raises.

    The source ``(9, 9)`` km lies outside the 4 km x 4 km mesh: its nearest
    node is farther than the largest cell diameter (``sqrt(2)`` km), the
    bound used by `Eikonal_Modeling.eikonal_bcs` to detect sources outside
    the mesh.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="no mesh node"):
        Eikonal_Modeling(2, [(9.0, 9.0)], degree_eik=1).eikonal_bcs(
            square_space(), lmin=1.0
        )


@pytest.mark.parametrize(
    "velocity, sources, expected",
    [
        (
            lambda x: 1.5 + 0.0 * x[0],
            [(2.2, 0.9)],
            [[2.0, 0.0], [2.0, 1.0], [3.0, 1.0]],
        ),
        (
            lambda x: 1.0 + x[0] / 4.0,
            [(2.2, 0.9)],
            [[2.0, 0.0], [2.0, 1.0], [3.0, 1.0]],
        ),
        (
            lambda x: 1.5 + 0.0 * x[0],
            [(1.0, 1.0), (2.0, 1.0)],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ],
        ),
    ],
    ids=["homogeneous", "linear_velocity", "overlapping_sources"],
)
def test_eikonal_bcs_local_time(
    velocity, sources: list[tuple[float, float]], expected: list[list[float]]
) -> None:
    """Check the local travel time on the nodes within ``lmin`` of the sources.

    On a P1 mesh with nodes at integer coordinates and ``lmin = 1``, every
    node within 1 km of a source is fixed to the smallest local travel time
    ``||x - s|| (1 / c(s) + 1 / c(x)) / 2`` over the sources, owned and halo
    copies alike, and no other node is touched. The velocity is linear, so its
    P1 interpolant and its value at the source are exact.

    Parameters
    ----------
    velocity : `callable`
        Velocity as a function of the coordinates, valid both for UFL
        coordinates and for numpy arrays.
    sources : `list`
        Source coordinates.
    expected : `list`
        Coordinates of the nodes that must hold the Dirichlet condition.

    Returns
    -------
    None
    """
    V = square_space()
    mesh = V.mesh()
    c = fire.Function(V).interpolate(velocity(fire.SpatialCoordinate(mesh)))
    eik = Eikonal_Modeling(2, sources, degree_eik=1, source_bc="local_time")
    bcs, marker = eik.eikonal_bcs(V, lmin=1.0, c=c)

    assert marked_points(marker) == expected

    fixed = fire.Function(V).assign(-1.0)
    for bc in bcs:
        bc.apply(fixed)
    coords = fire.assemble(
        fire.interpolate(
            fire.SpatialCoordinate(mesh),
            fire.VectorFunctionSpace(mesh, V.ufl_element()),
        )
    )
    times = [
        min(
            0.5
            * np.hypot(*(p - np.asarray(s)))
            * (1.0 / velocity(s) + 1.0 / velocity(p))
            for s in sources
        )
        for p in coords.dat.data_ro_with_halos
    ]
    marked = marker.dat.data_ro_with_halos == 1.0
    np.testing.assert_allclose(
        fixed.dat.data_ro_with_halos, np.where(marked, times, -1.0), rtol=1e-12
    )


def test_eikonal_bcs_local_time_needs_velocity() -> None:
    """Check that the 'local_time' source condition requires a velocity Function.

    Returns
    -------
    None
    """
    eik = Eikonal_Modeling(2, [(2.2, 0.9)], degree_eik=1, source_bc="local_time")
    with pytest.raises(TypeError, match="'c' must be of type"):
        eik.eikonal_bcs(square_space(), lmin=1.0)
    with pytest.raises(TypeError, match="'c' must be of type"):
        eik.eikonal_bcs(square_space(), lmin=1.0, c=1.5)


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


def eikonal_wave(
    geometry: str,
    degree: int,
    f_est: float,
    element_type: str,
    source_bc_eik: str = "nearest_node",
    dimension: int = 2,
    edge_length: float = 0.1,
) -> AcousticWave:
    """Run an Eikonal analysis and return the wave that holds its results.

    The setup is that of ``test_eikonal_new`` (unit square or cube, source at
    ``(-0.5, 0.25)`` or ``(-0.5, 0.25, 0.5)``) with the smooth velocity
    ``c = 3 - 1.5 x`` interpolated straight into the continuous space.
    Evaluating a discontinuous or DG0 velocity at nodes shared by several
    cells, or at cell centroids lying on the discontinuity, depends on the
    partition and on round-off, which would hide the behaviour under test.
    With the discontinuous velocity of ``test_eikonal_new``, a mesh with an
    odd number of elements N puts cell centroids exactly on x = 0.5: there
    the 'local_time' T-3D consistent case (N = 5) gives 83.563 ms with
    Firedrake 2025.4.1 and 84.164 ms with 2026.10.0.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".
    degree : `int`
        Finite element order for the Eikonal equation.
    f_est : `float`
        Factor for the stabilizing term in the Eikonal equation.
    element_type : `str`
        Finite element type for the Eikonal equation, "consistent" or
        "underintegrated" (KMV on triangles, spectral on quadrilaterals).
    source_bc_eik : `str`, optional
        Dirichlet condition at the sources, "nearest_node" or "local_time".
        Default is "nearest_node".
    dimension : `int`, optional
        Dimension of the problem, 2 or 3. Default is 2.
    edge_length : `float`, optional
        Mesh size in km. Default is 0.1.

    Returns
    -------
    `AcousticWave`
        Wave after the preamble mesh operations and the identification of the
        critical boundary points.
    """
    dictionary = wave_dict(geometry, dimension, degree, element_type, source_bc_eik)
    dictionary["visualization"]["output_folder"] += f"_np{fire.COMM_WORLD.size}"
    wave = AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    wave.set_initial_velocity_model(
        conditional=3.0 - 1.5 * wave.mesh_x, dg_velocity_model=False
    )
    wave.mesh_ops.preamble_mesh_operations(
        wave, ele_type_eik=element_type, f_est=f_est, source_bc_eik=source_bc_eik
    )
    wave.layer_ops.critical_boundary_points(wave)
    return wave


def critical_boundary_data(
    geometry: str,
    degree: int,
    f_est: float,
    element_type: str,
    source_bc_eik: str = "nearest_node",
    dimension: int = 2,
    edge_length: float = 0.1,
) -> list:
    """Run an Eikonal analysis and return its boundary data.

    The setup is that of `eikonal_wave`.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".
    degree : `int`
        Finite element order for the Eikonal equation.
    f_est : `float`
        Factor for the stabilizing term in the Eikonal equation.
    element_type : `str`
        Finite element type for the Eikonal equation, "consistent" or
        "underintegrated" (KMV on triangles, spectral on quadrilaterals).
    source_bc_eik : `str`, optional
        Dirichlet condition at the sources, "nearest_node" or "local_time".
        Default is "nearest_node".
    dimension : `int`, optional
        Dimension of the problem, 2 or 3. Default is 2.
    edge_length : `float`, optional
        Mesh size in km. Default is 0.1.

    Returns
    -------
    `list`
        ``[pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]`` per boundary,
        sorted by the minimum Eikonal.
    """
    wave = eikonal_wave(
        geometry, degree, f_est, element_type, source_bc_eik, dimension, edge_length
    )
    return wave.layer_ops.eik_bnd


# Serial results of critical_boundary_data:
# (critical point, minimum Eikonal in s) per absorbing boundary
@pytest.mark.parametrize(
    "geometry, degree, f_est, element_type, serial",
    [
        (
            "T",
            2,
            0.06,
            "consistent",
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
            "consistent",
            [
                ([-0.5, 0.0], 0.08929751297228099),
                ([-1.0, 0.15000000000000002], 0.1928097578035655),
                ([-0.5, 1.0], 0.37804633159604933),
            ],
        ),
        (
            "T",
            2,
            0.06,
            "underintegrated",
            [
                ([-0.5, 0.0], 0.08603482951166294),
                ([-1.0, 0.0], 0.17571950059880503),
                ([-0.45, 1.0], 0.3766437303516901),
            ],
        ),
        (
            "Q",
            2,
            0.05,
            "underintegrated",
            [
                ([-0.5, 0.0], 0.0660405994505825),
                ([-1.0, 0.15000000000000002], 0.18864019575613442),
                ([-0.5, 1.0], 0.3759377789296918),
            ],
        ),
    ],
)
def test_eikonal_critical_points_match_serial(
    geometry: str, degree: int, f_est: float, element_type: str, serial: list
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
    element_type : `str`
        Finite element type for the Eikonal equation, "consistent" or
        "underintegrated".
    serial : `list`
        Serial ``(pnt_crit, eikmin)`` per boundary, sorted by ``eikmin``.

    Returns
    -------
    None

    Notes
    -----
    The serial values match Firedrake 2026.10.0: two ranks of it reproduce
    them to the printed precision (T, consistent: 88.279 ms). A serial run and
    a two-rank run of the same installation agree to round-off, but
    installations differ by up to 4.1e-5 relative (T, consistent: 0.0882819 s
    with Firedrake 2025.4.1 instead of 0.0882785 s), so the minimum Eikonal is
    compared with ``rtol = 1e-4``. The critical points are compared exactly,
    and every rank must hold the same result.
    """
    eik_bnd = critical_boundary_data(geometry, degree, f_est, element_type)

    result = [(list(pnt), eikmin) for pnt, _, eikmin, *_ in eik_bnd]
    assert [pnt for pnt, _ in result] == [pnt for pnt, _ in serial]
    np.testing.assert_allclose(
        [e for _, e in result], [e for _, e in serial], rtol=1e-4
    )
    assert fire.COMM_WORLD.allgather(result) == [result] * fire.COMM_WORLD.size


def test_velocity_ranges_agree_on_all_ranks() -> None:
    """Check that the velocity ranges of the preamble are global.

    The velocity ``c = 3 - 1.5 x`` spans 1.5 to 3 km/s in the domain and on
    its absorbing boundaries, while each rank of a partition only holds part
    of it. The ranges set the velocity floor of the Eikonal equation and the
    size of the absorbing layer, so every rank must use the global values.

    Returns
    -------
    None

    Notes
    -----
    With ranges taken from the local data, two ranks found 2.1 km/s and
    1.5 km/s as minimum velocities (see
    `test_eikonal_kernels_agree_on_all_ranks`).
    """
    wave = eikonal_wave("Q", 2, 0.05, "underintegrated")

    ranges = (wave.c_min, wave.c_max, wave.c_bnd_min, wave.c_bnd_max)
    np.testing.assert_allclose(ranges, (1.5, 3.0, 1.5, 3.0), rtol=1e-12)
    assert fire.COMM_WORLD.allgather(ranges) == [ranges] * fire.COMM_WORLD.size


def test_mesh_sizes_agree_on_all_ranks() -> None:
    """Check that the representative mesh sizes are global.

    The unit square is meshed with 8 x 8 quadrilaterals and graded by
    ``(z, x) -> (z**2, x**2)``, so the smallest cell, a square of side 1/64,
    is at one corner and the largest, of side 15/64, at the opposite one, on
    different ranks. ``lmin`` and ``lmax`` are the cell diameters divided by
    ``sqrt(2)``, so they are the sides of these squares on every rank.

    Returns
    -------
    None

    Notes
    -----
    ``lmin`` sets the search tolerance and the source ball of the Eikonal
    boundary conditions, and both sizes enter the size of the absorbing layer,
    so every rank must use the same values.
    """
    mesh = fire.UnitSquareMesh(8, 8, quadrilateral=True)
    z, x = fire.SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(fire.as_vector([z**2, x**2]))
    mesh_ops = MeshOps((1.0, 1.0), dimension=2, quadrilateral=True)

    _, lmin, lmax, alpha, tol = mesh_ops.representative_mesh_dimensions(
        mesh, fire.FunctionSpace(mesh, "DG", 0)
    )

    sizes = (lmin, lmax, alpha, tol)
    np.testing.assert_allclose((lmin, lmax), (1.0 / 64, 15.0 / 64), atol=1e-6)
    assert fire.COMM_WORLD.allgather(sizes) == [sizes] * fire.COMM_WORLD.size


def test_eikonal_kernels_agree_on_all_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that every rank compiles the same Eikonal kernels.

    PyOP2 generates the code of a kernel on one rank and broadcasts it, so
    the ranks must build the same kernels. With ``spmd_strict``, PyOP2
    compares the cache keys of every rank before each lookup and raises
    ``ValueError`` if they differ, whatever the caches hold.

    Parameters
    ----------
    monkeypatch : `pytest.MonkeyPatch`
        Fixture that enables ``spmd_strict`` for this test only.

    Returns
    -------
    None

    Notes
    -----
    A velocity floor taken from the local data of each rank and written into
    the forms as a number made the ranks compile different kernels. With
    empty caches, the rank that did not generate the code then ran the floor
    of the other one: the minimum Eikonal on Xmax of this case was 319.833 ms
    instead of 375.938 ms, and the wrong kernel stayed in the disk cache for
    later serial runs.
    """
    monkeypatch.setitem(configuration, "spmd_strict", True)

    eik_bnd = critical_boundary_data("Q", 2, 0.05, "underintegrated")

    np.testing.assert_allclose(
        [eikmin for _, _, eikmin, *_ in eik_bnd],
        [0.0660405994505825, 0.18864019575613442, 0.3759377789296918],
        rtol=1e-4,
    )


@pytest.mark.parametrize("geometry", ["T", "Q"])
def test_local_time_eikonal_matches_theory_on_all_ranks(geometry: str) -> None:
    """Check the Eikonal with the 'local_time' source condition in parallel.

    Theoretical value. The velocity ``c = 3 - 1.5 x`` depends on x only, so
    the ray from the source ``(-0.5, 0.25)`` to the boundary ``x = 0`` keeps
    its z coordinate (the ray parameter along z is zero) and its travel time
    is

        t = int_0^0.25 dx / (3 - 1.5 x) = ln(3 / 2.625) / 1.5 = 89.021 ms,

    reached at ``(-0.5, 0)``. Every other absorbing boundary is at least
    0.5 km away through velocities of at most 3 km/s, i.e. at least 167 ms.

    The minimum Eikonal must match it closely, be the same on every rank,
    and barely depend on the stabilizing factor, unlike with the
    'nearest_node' condition (2D, degree 2, consistent, edge length 0.1 km).
    Measured: T 89.451 / 89.462 ms and Q 89.247 / 89.268 ms for f_est 0.05 /
    0.06, identical on one and two ranks with Firedrake 2025.4.1 and the same
    on two ranks with Firedrake 2026.10.0.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".

    Returns
    -------
    None
    """
    theory = np.log(3.0 / (3.0 - 1.5 * 0.25)) / 1.5
    eikmins = []
    for f_est in (0.05, 0.06):
        eik_bnd = critical_boundary_data(
            geometry, 2, f_est, "consistent", source_bc_eik="local_time"
        )
        result = [(list(pnt), eikmin) for pnt, _, eikmin, *_ in eik_bnd]
        assert fire.COMM_WORLD.allgather(result) == [result] * fire.COMM_WORLD.size
        eikmins.append(result[0][1])

    np.testing.assert_allclose(eikmins, theory, rtol=6e-3)
    assert abs(eikmins[1] - eikmins[0]) < 1e-3 * theory


@pytest.mark.parametrize(
    "geometry, edge_length, f_est",
    [("T", 1.0 / 12, 0.02), ("Q", 1.0 / 6, 0.09)],
)
def test_local_time_eikonal_3d_matches_theory_on_all_ranks(
    geometry: str, edge_length: float, f_est: float
) -> None:
    """Check the 3D Eikonal with the 'local_time' source condition in parallel.

    Theoretical value. The velocity ``c = 3 - 1.5 x`` depends on x only, so
    the ray from the source ``(-0.5, 0.25, 0.5)`` to the boundary ``x = 0``
    keeps its z and y coordinates (the ray parameters along z and y are zero)
    and its travel time is

        t = int_0^0.25 dx / (3 - 1.5 x) = ln(3 / 2.625) / 1.5 = 89.021 ms,

    reached at ``(-0.5, 0, 0.5)``. Every other absorbing boundary is at least
    0.5 km away through velocities of at most 3 km/s, i.e. at least 167 ms.

    Calibration (degree 2, consistent, Firedrake 2025.4.1). Unlike in 2D, the
    3D result depends on f_est, roughly in proportion to it (the artificial
    diffusion acts on the point-source cone, whose Laplacian (d - 1) / r is
    twice as large in 3D), so the stabilizing factor must be small. Minimum
    Eikonal error for N = 1 / edge_length and f_est (r: the solver retried
    with a larger f_est):

        T  N    f=0.01      f=0.02   f=0.03   f=0.04   f=0.05   f=0.06
           7    r (0.04)    r        r        +2.66 %  +3.00 %  +3.15 %
           8    r (0.02)    +2.70 %  +4.18 %  +5.20 %  +6.00 %  +6.71 %
           10   r (0.02)    +2.00 %  +3.64 %  +4.76 %  +5.68 %  +6.52 %
           12   r (0.02)    +0.70 %* +3.47 %  +5.11 %  +6.42 %  +7.61 %

        Q  N    f=0.05      f=0.07   f=0.08   f=0.09   f=0.10
           5    -7.05 %     -8.97 %  -9.72 %  -10.4 %  -10.9 %
           6    -1.17 %     -0.51 %  -0.29 %  -0.10 %* +0.07 %

    T: N = 7 with f_est = 0.05 is the coarsest stable choice below 3 %, but
    at +2.999 % it leaves no margin, so N = 12 with f_est = 0.02 (+0.70 %) is
    used; below 0.02 the solver needs more stabilization and retries. Q: on
    N = 5 the source region (radius lmin = 0.2 km) covers most of the
    0.25 km to the boundary and the result falls as f_est grows, so N = 6
    with f_est = 0.09 (-0.10 %, stable at 0.08 and 0.10) is used. One and
    two ranks give the same values (T: to 4e-6 relative, Q: identical).
    Firedrake 2026.10.0 on two ranks gives T 89.644 ms (+0.70 %) and
    Q 88.936 ms (-0.10 %), within 3e-5 of the 2025.4.1 values. Both
    configurations converge at their f_est without retries with both
    versions.

    Parameters
    ----------
    geometry : `str`
        Element geometry, "T" or "Q".
    edge_length : `float`
        Mesh size in km.
    f_est : `float`
        Factor for the stabilizing term in the Eikonal equation.

    Returns
    -------
    None
    """
    theory = np.log(3.0 / (3.0 - 1.5 * 0.25)) / 1.5
    eik_bnd = critical_boundary_data(
        geometry,
        2,
        f_est,
        "consistent",
        source_bc_eik="local_time",
        dimension=3,
        edge_length=edge_length,
    )
    result = [(list(pnt), eikmin) for pnt, _, eikmin, *_ in eik_bnd]

    assert fire.COMM_WORLD.allgather(result) == [result] * fire.COMM_WORLD.size
    np.testing.assert_allclose(result[0][0], [-0.5, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(result[0][1], theory, rtol=1e-2)
