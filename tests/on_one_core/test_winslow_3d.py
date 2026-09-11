from pathlib import Path

import numpy as np
import pytest

from spyro.io.basicio import parallel_print
from spyro.meshing.meshing_utils3D import create_sizing_function3D
from spyro.meshing.meshing_winslow3D import (
    NUMBA_AVAILABLE,
    _numba_winslow_3d_step,
    _resolve_stencil_coordinates,
    run_selected_winslow,
    winslow_smooth_3d55,
)


VS_EXAMPLE_3D = Path("velocity_models/vs_example_3D.bin")


def _structured_hex_mesh(
    nx=3,
    ny=3,
    nz=3,
    length_x=2.0,
    length_y=2.0,
    depth_z=2.0,
):
    """Create a structured hexahedral mesh in Spyro coordinates."""
    x_values = np.linspace(0.0, length_x, nx)
    y_values = np.linspace(0.0, length_y, ny)
    z_values = np.linspace(0.0, -depth_z, nz)

    points = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                points.append(
                    [
                        x_values[i],
                        y_values[j],
                        z_values[k],
                    ]
                )

    points = np.asarray(points, dtype=float)

    def node(i, j, k):
        return i + nx * (j + ny * k)

    hexes = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                hexes.append(
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]
                )

    return points, np.asarray(hexes, dtype=np.int64), node


def _constant_sizing(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim == 1:
        return 1.0
    return np.ones(coordinates.shape[0], dtype=float)


def _varying_sizing(coordinates):
    """Positive sizing law in (z, x, y) coordinates."""
    coordinates = np.asarray(coordinates, dtype=float)
    scalar = coordinates.ndim == 1
    coordinates = np.atleast_2d(coordinates)

    z = coordinates[:, 0]
    x = coordinates[:, 1]
    y = coordinates[:, 2]

    values = 1.5 + 0.14 * x + 0.09 * y - 0.07 * z

    if scalar:
        return float(values[0])
    return values


def _python_implementation(function):
    """Return the original Python implementation of a Numba dispatcher."""
    return getattr(function, "py_func", function)


def test_resolve_stencil_coordinates_complete_neighbors():
    function = _python_implementation(_resolve_stencil_coordinates)

    coordinates = np.arange(30, dtype=float)
    center = 100.0
    stencil = np.arange(1, 19, dtype=np.int32)

    resolved = function(coordinates, center, stencil)

    assert len(resolved) == 18
    np.testing.assert_allclose(
        resolved,
        coordinates[1:19],
    )


def test_resolve_stencil_coordinates_reflects_missing_cardinals():
    function = _python_implementation(_resolve_stencil_coordinates)

    coordinates = np.array(
        [
            0.0,
            -2.0,   # west
            -3.0,   # south
            -4.0,   # bottom
        ],
        dtype=float,
    )

    center = 1.0
    stencil = np.full(18, -1, dtype=np.int32)
    stencil[1] = 1  # W
    stencil[3] = 2  # S
    stencil[5] = 3  # B

    resolved = function(coordinates, center, stencil)

    east, west, north, south, top, bottom = resolved[:6]

    assert east == pytest.approx(4.0)
    assert west == pytest.approx(-2.0)
    assert north == pytest.approx(5.0)
    assert south == pytest.approx(-3.0)
    assert top == pytest.approx(6.0)
    assert bottom == pytest.approx(-4.0)

    ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb = resolved[6:]

    assert ne == pytest.approx(north + east - center)
    assert nw == pytest.approx(north + west - center)
    assert se == pytest.approx(south + east - center)
    assert sw == pytest.approx(south + west - center)
    assert nt == pytest.approx(north + top - center)
    assert nb == pytest.approx(north + bottom - center)
    assert st == pytest.approx(south + top - center)
    assert sb == pytest.approx(south + bottom - center)
    assert et == pytest.approx(east + top - center)
    assert eb == pytest.approx(east + bottom - center)
    assert wt == pytest.approx(west + top - center)
    assert wb == pytest.approx(west + bottom - center)


def test_resolve_stencil_coordinates_all_missing_returns_center():
    function = _python_implementation(_resolve_stencil_coordinates)

    coordinates = np.array([7.0], dtype=float)
    center = 7.0
    stencil = np.full(18, -1, dtype=np.int32)

    resolved = function(coordinates, center, stencil)

    np.testing.assert_allclose(
        resolved,
        np.full(18, center),
    )


# -----------------------------------------------------------------------------
# _numba_winslow_3d_step
# -----------------------------------------------------------------------------


def _kernel_stencil_problem():
    """Create one complete local 3-D stencil around a center node."""
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],    # center
            [1.0, 0.0, 0.0],    # E
            [-1.0, 0.0, 0.0],   # W
            [0.0, 1.0, 0.0],    # N
            [0.0, -1.0, 0.0],   # S
            [0.0, 0.0, 1.0],    # T
            [0.0, 0.0, -1.0],   # B
            [1.0, 1.0, 0.0],    # NE
            [-1.0, 1.0, 0.0],   # NW
            [1.0, -1.0, 0.0],   # SE
            [-1.0, -1.0, 0.0],  # SW
            [0.0, 1.0, 1.0],    # NT
            [0.0, 1.0, -1.0],   # NB
            [0.0, -1.0, 1.0],   # ST
            [0.0, -1.0, -1.0],  # SB
            [1.0, 0.0, 1.0],    # ET
            [1.0, 0.0, -1.0],   # EB
            [-1.0, 0.0, 1.0],   # WT
            [-1.0, 0.0, -1.0],  # WB
        ],
        dtype=float,
    )

    stencils = np.full(
        (len(points), 18),
        -1,
        dtype=np.int32,
    )
    stencils[0] = np.arange(1, 19, dtype=np.int32)

    sizing_values = (
        2.0
        + 0.18 * points[:, 0]
        + 0.11 * points[:, 1]
        + 0.07 * points[:, 2]
    )

    is_movable = np.ones(len(points), dtype=np.bool_)

    move_x = np.zeros(len(points), dtype=np.bool_)
    move_y = np.zeros(len(points), dtype=np.bool_)
    move_z = np.zeros(len(points), dtype=np.bool_)

    move_x[0] = True
    move_y[0] = True
    move_z[0] = True

    return (
        points,
        sizing_values,
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
    )


def test_numba_step_python_path_updates_only_movable_center():
    function = _python_implementation(_numba_winslow_3d_step)

    (
        points,
        sizing_values,
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
    ) = _kernel_stencil_problem()

    x_new, y_new, z_new = function(
        np.ascontiguousarray(points[:, 0]),
        np.ascontiguousarray(points[:, 1]),
        np.ascontiguousarray(points[:, 2]),
        np.ascontiguousarray(sizing_values),
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
        0.5,
    )

    result = np.column_stack((x_new, y_new, z_new))

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[1:], points[1:])

    # The non-uniform sizing field produces a movement of the center.
    assert np.linalg.norm(result[0] - points[0]) > 0.0


def test_numba_step_python_path_skips_fixed_nodes():
    function = _python_implementation(_numba_winslow_3d_step)

    (
        points,
        sizing_values,
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
    ) = _kernel_stencil_problem()

    move_x[:] = False
    move_y[:] = False
    move_z[:] = False

    result = function(
        np.ascontiguousarray(points[:, 0]),
        np.ascontiguousarray(points[:, 1]),
        np.ascontiguousarray(points[:, 2]),
        np.ascontiguousarray(sizing_values),
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
        0.5,
    )

    np.testing.assert_allclose(result[0], points[:, 0])
    np.testing.assert_allclose(result[1], points[:, 1])
    np.testing.assert_allclose(result[2], points[:, 2])


def test_numba_step_python_path_handles_degenerate_denominator():
    function = _python_implementation(_numba_winslow_3d_step)

    num_nodes = 7

    x = np.zeros(num_nodes, dtype=float)
    y = np.zeros(num_nodes, dtype=float)
    z = np.zeros(num_nodes, dtype=float)
    sizing = np.ones(num_nodes, dtype=float)

    stencils = np.full(
        (num_nodes, 18),
        -1,
        dtype=np.int32,
    )
    stencils[0, :6] = np.arange(1, 7, dtype=np.int32)

    is_movable = np.ones(num_nodes, dtype=np.bool_)
    move_x = np.zeros(num_nodes, dtype=np.bool_)
    move_y = np.zeros(num_nodes, dtype=np.bool_)
    move_z = np.zeros(num_nodes, dtype=np.bool_)

    move_x[0] = True
    move_y[0] = True
    move_z[0] = True

    x_new, y_new, z_new = function(
        x,
        y,
        z,
        sizing,
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
        0.5,
    )

    np.testing.assert_allclose(x_new, x)
    np.testing.assert_allclose(y_new, y)
    np.testing.assert_allclose(z_new, z)


@pytest.mark.skipif(
    not NUMBA_AVAILABLE,
    reason="Numba is not installed in this environment.",
)
def test_numba_step_compiled_path_executes():
    (
        points,
        sizing_values,
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
    ) = _kernel_stencil_problem()

    result = _numba_winslow_3d_step(
        np.ascontiguousarray(points[:, 0]),
        np.ascontiguousarray(points[:, 1]),
        np.ascontiguousarray(points[:, 2]),
        np.ascontiguousarray(sizing_values),
        stencils,
        is_movable,
        move_x,
        move_y,
        move_z,
        0.25,
    )

    assert len(result) == 3
    for values in result:
        assert values.shape == (len(points),)
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    "bad_points",
    [
        np.zeros(3),
        np.zeros((4, 2)),
        np.zeros((2, 2, 3)),
    ],
)
def test_winslow_smooth_rejects_invalid_point_shape(bad_points):
    _, hexes, _ = _structured_hex_mesh()

    with pytest.raises(
        ValueError,
        match="points must have shape",
    ):
        winslow_smooth_3d55(
            points=bad_points,
            hexes=hexes,
            move_all={0},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            comm=None,
            parallel_print=parallel_print,
            iterations=1,
            omega=0.5,
        )


@pytest.mark.parametrize(
    "bad_hexes",
    [
        np.zeros(8, dtype=np.int64),
        np.zeros((1, 7), dtype=np.int64),
        np.zeros((1, 9), dtype=np.int64),
    ],
)
def test_winslow_smooth_rejects_invalid_hex_shape(bad_hexes):
    points, _, _ = _structured_hex_mesh()

    with pytest.raises(
        ValueError,
        match="hexes must have shape",
    ):
        winslow_smooth_3d55(
            points=points,
            hexes=bad_hexes,
            move_all={13},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            comm=None,
            parallel_print=parallel_print,
            iterations=1,
            omega=0.5,
        )


def test_winslow_smooth_rejects_negative_iterations():
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    with pytest.raises(
        ValueError,
        match="iterations must be non-negative",
    ):
        winslow_smooth_3d55(
            points=points,
            hexes=hexes,
            move_all={center},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            comm=None,
            parallel_print=parallel_print,
            iterations=-1,
            omega=0.5,
        )


@pytest.mark.parametrize("omega", [0.0, -0.1, 1.01, 2.0])
def test_winslow_smooth_rejects_invalid_omega(omega):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    with pytest.raises(
        ValueError,
        match="omega must satisfy",
    ):
        winslow_smooth_3d55(
            points=points,
            hexes=hexes,
            move_all={center},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            comm=None,
            parallel_print=parallel_print,
            iterations=1,
            omega=omega,
        )


@pytest.mark.parametrize("invalid_node", [-1, 27, 100])
def test_winslow_smooth_rejects_invalid_movement_indices(invalid_node):
    points, hexes, _ = _structured_hex_mesh()

    with pytest.raises(
        ValueError,
        match="Movement sets contain node indices outside the mesh",
    ):
        winslow_smooth_3d55(
            points=points,
            hexes=hexes,
            move_all={invalid_node},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            comm=None,
            parallel_print=parallel_print,
            iterations=0,
            omega=0.5,
        )


def test_winslow_smooth_zero_iterations_preserves_real_hex_mesh(capsys):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    smoothed = winslow_smooth_3d55(
        points=points,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_varying_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        comm=None,
        parallel_print=parallel_print,
        iterations=0,
        omega=0.5,
    )

    np.testing.assert_allclose(smoothed, points)

    output = capsys.readouterr().out
    assert "Smoothing Complete." in output


def test_winslow_smooth_isolated_movable_node_is_safe():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    hexes = np.empty((0, 8), dtype=np.int64)

    smoothed = winslow_smooth_3d55(
        points=points,
        hexes=hexes,
        move_all={0},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_constant_sizing,
        length_x=1.0,
        length_y=1.0,
        depth_z=1.0,
        comm=None,
        parallel_print=parallel_print,
        iterations=1,
        omega=0.5,
    )

    np.testing.assert_allclose(smoothed, points)


@pytest.mark.parametrize(
    "movement_set",
    ["X", "Y", "Z"],
)
def test_winslow_smooth_respects_directional_movement(movement_set):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    distorted = points.copy()
    distorted[center] += np.array([0.18, -0.14, 0.11])

    move_x = {center} if movement_set == "X" else set()
    move_y = {center} if movement_set == "Y" else set()
    move_z = {center} if movement_set == "Z" else set()

    smoothed = winslow_smooth_3d55(
        points=distorted,
        hexes=hexes,
        move_all=set(),
        move_X_only=move_x,
        move_Y_only=move_y,
        move_Z_only=move_z,
        ef_segy=_varying_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        comm=None,
        parallel_print=parallel_print,
        iterations=2,
        omega=0.5,
    )

    # Every non-movable node must remain fixed.
    fixed = np.arange(len(points)) != center
    np.testing.assert_allclose(smoothed[fixed], distorted[fixed])

    delta = smoothed[center] - distorted[center]

    if movement_set == "X":
        assert abs(delta[0]) > 1.0e-12
        assert delta[1] == pytest.approx(0.0)
        assert delta[2] == pytest.approx(0.0)
    elif movement_set == "Y":
        assert delta[0] == pytest.approx(0.0)
        assert abs(delta[1]) > 1.0e-12
        assert delta[2] == pytest.approx(0.0)
    else:
        assert delta[0] == pytest.approx(0.0)
        assert delta[1] == pytest.approx(0.0)
        assert abs(delta[2]) > 1.0e-12


def test_winslow_smooth_move_all_changes_distorted_interior():
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    distorted = points.copy()
    distorted[center] += np.array([0.16, -0.12, 0.09])

    smoothed = winslow_smooth_3d55(
        points=distorted,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_varying_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        comm=None,
        parallel_print=parallel_print,
        iterations=3,
        omega=0.4,
    )

    assert np.all(np.isfinite(smoothed))
    assert np.linalg.norm(
        smoothed[center] - distorted[center]
    ) > 1.0e-12

    fixed = np.arange(len(points)) != center
    np.testing.assert_allclose(smoothed[fixed], distorted[fixed])


class _RankOneCommunicator:
    rank = 1


class _CommunicatorWrapper:
    comm = _RankOneCommunicator()


def test_winslow_smooth_accepts_wrapped_communicator():
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    result = winslow_smooth_3d55(
        points=points,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_constant_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        comm=_CommunicatorWrapper(),
        parallel_print=parallel_print,
        iterations=1,
        omega=0.5,
    )

    assert result.shape == points.shape
    assert np.all(np.isfinite(result))


def test_run_selected_winslow_rejects_empty_movable_union():
    points, hexes, _ = _structured_hex_mesh()

    with pytest.raises(
        RuntimeError,
        match="Winslow smoothing has no movable nodes",
    ):
        run_selected_winslow(
            points=points,
            hexes=hexes,
            move_all=set(),
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            winslow_iterations=1,
            winslow_omega=0.5,
            comm=None,
            parallel_print=parallel_print,
            selected_winslow="numba",
        )


def test_run_selected_winslow_rejects_unknown_implementation(capsys):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    with pytest.raises(
        ValueError,
        match="Only selected_winslow='numba'",
    ):
        run_selected_winslow(
            points=points,
            hexes=hexes,
            move_all={center},
            move_X_only=set(),
            move_Y_only=set(),
            move_Z_only=set(),
            ef_segy=_constant_sizing,
            length_x=2.0,
            length_y=2.0,
            depth_z=2.0,
            winslow_iterations=0,
            winslow_omega=0.5,
            comm=None,
            parallel_print=parallel_print,
            selected_winslow="python",
        )

    output = capsys.readouterr().out
    assert "effectively constant" in output


def test_run_selected_winslow_real_algorithm_reports_displacement(capsys):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    distorted = points.copy()
    distorted[center] += np.array([0.15, -0.10, 0.08])

    result = run_selected_winslow(
        points=distorted,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_varying_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        winslow_iterations=2,
        winslow_omega=0.5,
        comm=None,
        parallel_print=parallel_print,
        selected_winslow="numba",
    )

    assert result.shape == distorted.shape
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result[center] - distorted[center]) > 1.0e-12

    output = capsys.readouterr().out
    assert "Winslow sizing on movable nodes" in output
    assert "Winslow movable-node counts" in output
    assert "Winslow displacement" in output


def test_run_selected_winslow_zero_iterations_reports_no_movement(capsys):
    points, hexes, node = _structured_hex_mesh()
    center = node(1, 1, 1)

    result = run_selected_winslow(
        points=points,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=_varying_sizing,
        length_x=2.0,
        length_y=2.0,
        depth_z=2.0,
        winslow_iterations=0,
        winslow_omega=0.5,
        comm=None,
        parallel_print=parallel_print,
        selected_winslow="numba",
    )

    np.testing.assert_allclose(result, points)

    output = capsys.readouterr().out
    assert "completed but no movable node changed position" in output


@pytest.mark.slow
def test_run_selected_winslow_with_vs_example_3d():
    if not VS_EXAMPLE_3D.exists():
        pytest.skip(
            f"Required velocity model not found: {VS_EXAMPLE_3D}"
        )

    bbox = (
        -3000.0,
        0.0,
        0.0,
        5000.0,
        0.0,
        5000.0,
    )

    ef_segy, _, _, nz, nx, ny = create_sizing_function3D(
        fname=VS_EXAMPLE_3D,
        hmin=0.0,
        bbox=bbox,
        wl=2.0,
        freq=3.0,
        pad_type=None,
        pad_size_x=0.0,
        pad_size_y=0.0,
        pad_size_z=0.0,
        grade=0.05,
        vp_water=1000.0,
        nz=101,
        nx=101,
        ny=101,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float32",
    )

    assert (nz, nx, ny) == (101, 101, 101)

    points, hexes, node = _structured_hex_mesh(
        nx=3,
        ny=3,
        nz=3,
        length_x=5000.0,
        length_y=5000.0,
        depth_z=3000.0,
    )
    center = node(1, 1, 1)

    result = run_selected_winslow(
        points=points,
        hexes=hexes,
        move_all={center},
        move_X_only=set(),
        move_Y_only=set(),
        move_Z_only=set(),
        ef_segy=ef_segy,
        length_x=5000.0,
        length_y=5000.0,
        depth_z=3000.0,
        winslow_iterations=0,
        winslow_omega=0.5,
        comm=None,
        parallel_print=parallel_print,
        selected_winslow="numba",
    )

    np.testing.assert_allclose(result, points)
