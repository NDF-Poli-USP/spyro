from pathlib import Path

import numpy as np
import pytest

from spyro.meshing.meshing_utils3D import (
    _read_velocity_binary3D,
    create_sizing_function3D,
    define_winslow_points_3d,
    sizing_function_xyz,
)


VS_EXAMPLE_3D_BIN = Path("velocity_models/vs_example_3D.bin")
VS_EXAMPLE_3D_SHAPE = (101, 101, 101)
VS_EXAMPLE_3D_BBOX = (-3000.0, 0.0, 0.0, 5000.0, 0.0, 5000.0)


def _write_velocity_binary(
    filename,
    canonical_values,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
):
    """Write (z, x, y) values using Spyro."""
    canonical_values = np.asarray(canonical_values)
    inverse = np.argsort(np.asarray(axes_order, dtype=int))
    raw_values = canonical_values.transpose(tuple(inverse))

    endian = ">" if byte_order == "big" else "<"
    disk_dtype = np.dtype(dtype).newbyteorder(endian)
    np.asarray(raw_values, dtype=disk_dtype).ravel(
        order=axes_order_sort
    ).tofile(filename)


def _extract_gmsh_points(gmsh):
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    tag_to_index = {
        int(node_tag): node_index
        for node_index, node_tag in enumerate(node_tags)
    }
    return points, tag_to_index


@pytest.fixture
def gmsh_session():
    gmsh = pytest.importorskip("gmsh")

    if gmsh.isInitialized():
        gmsh.finalize()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("meshing_utils_3d_test")

    try:
        yield gmsh
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _build_water_and_subsurface_mesh(gmsh, water_group_name="Water"):
    """Build a small 3-D Gmsh model with water above subsurface."""
    occ = gmsh.model.occ

    subsoil = occ.addBox(0.0, 0.0, -8.0, 10.0, 12.0, 6.0)
    water = occ.addBox(0.0, 0.0, -2.0, 10.0, 12.0, 2.0)
    occ.synchronize()

    subsoil_group = gmsh.model.addPhysicalGroup(3, [subsoil])
    gmsh.model.setPhysicalName(3, subsoil_group, "Subsurface")
    water_group = gmsh.model.addPhysicalGroup(3, [water])
    gmsh.model.setPhysicalName(3, water_group, water_group_name)

    gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0)
    gmsh.model.mesh.generate(3)

    return _extract_gmsh_points(gmsh)


def test_sizing_function_xyz_projects_to_model_edges():
    received = []

    def sizing_field(points):
        received.append(np.asarray(points).copy())
        return 100.0 + 0.01 * points[:, 1] + 0.02 * points[:, 2]

    X = np.array([[-10.0, 2500.0], [5100.0, 1000.0]])
    Y = np.array([[-50.0, 2500.0], [5200.0, 4000.0]])
    Z = np.array([[100.0, -1500.0], [-4000.0, -500.0]])

    sizes = sizing_function_xyz(
        X,
        Y,
        Z,
        sizing_field,
        length_x=5000.0,
        length_y=5000.0,
        depth_z=3000.0,
    )

    expected_queries = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1500.0, 2500.0, 2500.0],
            [-3000.0, 5000.0, 5000.0],
            [-500.0, 1000.0, 4000.0],
        ]
    )

    assert sizes.shape == X.shape
    assert np.all(np.isfinite(sizes))
    assert np.all(sizes > 0.0)
    assert len(received) == 1
    assert np.allclose(received[0], expected_queries)


def test_sizing_function_xyz_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        sizing_function_xyz(
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            np.zeros((2, 2)),
            lambda points: np.ones(len(points)),
            1.0,
            1.0,
            1.0,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sizing_function_xyz_rejects_nonfinite_coordinates(bad_value):
    X = np.zeros((2, 2))
    X[0, 0] = bad_value

    with pytest.raises(FloatingPointError, match="invalid nodes"):
        sizing_function_xyz(
            X,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            lambda points: np.ones(len(points)),
            1.0,
            1.0,
            1.0,
        )


def test_sizing_function_xyz_rejects_nonfinite_sizes():
    def bad_sizing(points):
        values = np.ones(len(points))
        values[0] = np.nan
        return values

    with pytest.raises(ValueError, match="returned NaN or infinity"):
        sizing_function_xyz(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            bad_sizing,
            1.0,
            1.0,
            1.0,
        )


@pytest.mark.parametrize("bad_size", [0.0, -1.0])
def test_sizing_function_xyz_requires_positive_sizes(bad_size):
    with pytest.raises(ValueError, match="must be positive"):
        sizing_function_xyz(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            lambda points: np.full(len(points), bad_size),
            1.0,
            1.0,
            1.0,
        )


@pytest.mark.parametrize(
    "byte_order,axes_order,axes_order_sort",
    [
        ("big", (0, 1, 2), "F"),
        ("little", (1, 0, 2), "C"),
        ("big", (2, 0, 1), "F"),
    ],
)
def test_read_velocity_binary3d_round_trip(
    tmp_path,
    byte_order,
    axes_order,
    axes_order_sort,
):
    expected = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) + 1400.0
    filename = tmp_path / "velocity.bin"

    _write_velocity_binary(
        filename,
        expected,
        byte_order=byte_order,
        axes_order=axes_order,
        axes_order_sort=axes_order_sort,
    )

    actual = _read_velocity_binary3D(
        filename,
        nz=2,
        nx=3,
        ny=4,
        byte_order=byte_order,
        axes_order=axes_order,
        axes_order_sort=axes_order_sort,
        dtype="float32",
    )

    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_read_velocity_binary3d_validation(tmp_path):
    filename = tmp_path / "velocity.bin"
    np.ones(8, dtype=">f4").tofile(filename)

    with pytest.raises(FileNotFoundError, match="Velocity model not found"):
        _read_velocity_binary3D(tmp_path / "missing.bin", 2, 2, 2)

    with pytest.raises(ValueError, match="byte_order"):
        _read_velocity_binary3D(
            filename,
            2,
            2,
            2,
            byte_order="native",
        )

    with pytest.raises(ValueError, match="axes_order_sort"):
        _read_velocity_binary3D(
            filename,
            2,
            2,
            2,
            axes_order_sort="A",
        )

    with pytest.raises(ValueError, match="axes_order must be a permutation"):
        _read_velocity_binary3D(
            filename,
            2,
            2,
            2,
            axes_order=(0, 0, 2),
        )

    with pytest.raises(ValueError, match="expected 27"):
        _read_velocity_binary3D(filename, 3, 3, 3)


@pytest.mark.slow
def test_read_vs_example_3d_binary():
    if not VS_EXAMPLE_3D_BIN.exists():
        pytest.skip(f"Missing {VS_EXAMPLE_3D_BIN}")

    velocity = _read_velocity_binary3D(
        VS_EXAMPLE_3D_BIN,
        nz=101,
        nx=101,
        ny=101,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float32",
    )

    assert velocity.shape == VS_EXAMPLE_3D_SHAPE
    assert np.all(np.isfinite(velocity))
    assert np.max(velocity) > 0.0


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        ({"nz": None, "nx": 2, "ny": 2}, "requires nz, nx and ny"),
        ({"nz": 2, "nx": 2, "ny": 2, "wl": 0.0}, "wl must be positive"),
        ({"nz": 2, "nx": 2, "ny": 2, "freq": 0.0}, "freq must be positive"),
        (
            {
                "nz": 2,
                "nx": 2,
                "ny": 2,
                "bbox": (0.0, -1.0, 0.0, 1.0, 0.0, 1.0),
            },
            "bbox must be ordered",
        ),
    ],
)
def test_create_sizing_function3d_input_validation(tmp_path, kwargs, error_match):
    filename = tmp_path / "unused.bin"
    base = {
        "fname": filename,
        "hmin": 100.0,
        "bbox": (-1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        "wl": 2.0,
        "freq": 3.0,
        "nz": 2,
        "nx": 2,
        "ny": 2,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error_match):
        create_sizing_function3D(**base)


def test_create_sizing_function3d_fallback_interpolator(tmp_path):
    """Use float64 data so SeismicMesh's default float32 reader falls back."""
    velocity = np.linspace(1200.0, 3600.0, 3 * 4 * 5).reshape(3, 4, 5)
    filename = tmp_path / "velocity_float64.raw"

    _write_velocity_binary(
        filename,
        velocity,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float64",
    )

    ef, hmin, hmax, nz, nx, ny = create_sizing_function3D(
        fname=filename,
        hmin=250.0,
        bbox=(-2.0, 0.0, 0.0, 3.0, 0.0, 4.0),
        wl=2.0,
        freq=3.0,
        pad_type="rectangular",
        pad_size_x=1.0,
        pad_size_y=2.0,
        pad_size_z=3.0,
        grade=0.05,
        vp_water=None,
        nz=3,
        nx=4,
        ny=5,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float64",
    )

    assert hmax is not None
    assert (nz, nx, ny) == (3, 4, 5)
    assert hmin >= 250.0
    assert hmax >= hmin

    scalar_value = ef(np.array([-1.0, 1.5, 2.0]))
    vector_values = ef(
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 3.0, 4.0],
                [-10.0, -5.0, 20.0],
            ]
        )
    )
    edge_value = ef(np.array([-2.0, 0.0, 4.0]))

    assert np.isscalar(scalar_value)
    assert vector_values.shape == (3,)
    assert np.all(np.isfinite(vector_values))
    assert np.all(vector_values >= 250.0)
    assert vector_values[2] == pytest.approx(edge_value)

    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        ef(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="contain NaN or infinity"):
        ef(np.array([np.nan, 0.0, 0.0]))


def test_create_sizing_function3d_replaces_zero_velocity_with_water(tmp_path):
    velocity = np.full((2, 2, 2), 1800.0)
    velocity[0, 0, 0] = 0.0
    filename = tmp_path / "water_velocity_float64.raw"

    _write_velocity_binary(filename, velocity, dtype="float64")

    ef, hmin, hmax, nz, nx, ny = create_sizing_function3D(
        fname=filename,
        hmin=None,
        bbox=(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        wl=2.0,
        freq=3.0,
        vp_water=1500.0,
        nz=2,
        nx=2,
        ny=2,
        dtype="float64",
    )

    assert hmax is not None
    assert (nz, nx, ny) == (2, 2, 2)
    assert hmin == pytest.approx(250.0)
    assert hmax == pytest.approx(300.0)
    assert np.all(ef(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])) > 0.0)


@pytest.mark.parametrize("bad_value,error_match", [(0.0, "non-positive"), (-10.0, "non-positive")])
def test_create_sizing_function3d_rejects_nonpositive_velocity(
    tmp_path,
    bad_value,
    error_match,
):
    velocity = np.full((2, 2, 2), 1800.0)
    velocity[0, 0, 0] = bad_value
    filename = tmp_path / "bad_velocity_float64.raw"
    _write_velocity_binary(filename, velocity, dtype="float64")

    with pytest.raises(ValueError, match=error_match):
        create_sizing_function3D(
            fname=filename,
            hmin=None,
            bbox=(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
            wl=2.0,
            freq=3.0,
            vp_water=None,
            nz=2,
            nx=2,
            ny=2,
            dtype="float64",
        )


def test_create_sizing_function3d_rejects_nonfinite_velocity(tmp_path):
    velocity = np.full((2, 2, 2), 1800.0)
    velocity[0, 0, 0] = np.nan
    filename = tmp_path / "nan_velocity_float64.raw"
    _write_velocity_binary(filename, velocity, dtype="float64")

    with pytest.raises(ValueError, match="contains NaN or infinity"):
        create_sizing_function3D(
            fname=filename,
            hmin=None,
            bbox=(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
            wl=2.0,
            freq=3.0,
            vp_water=None,
            nz=2,
            nx=2,
            ny=2,
            dtype="float64",
        )


@pytest.mark.slow
def test_create_sizing_function3d_vs_example():
    if not VS_EXAMPLE_3D_BIN.exists():
        pytest.skip(f"Missing {VS_EXAMPLE_3D_BIN}")

    ef, hmin, hmax, nz, nx, ny = create_sizing_function3D(
        fname=VS_EXAMPLE_3D_BIN,
        hmin=150.0,
        bbox=VS_EXAMPLE_3D_BBOX,
        wl=2.0,
        freq=3.0,
        pad_type="rectangular",
        pad_size_x=1000.0,
        pad_size_y=1000.0,
        pad_size_z=1000.0,
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

    assert (nz, nx, ny) == VS_EXAMPLE_3D_SHAPE
    assert hmin >= 150.0
    if hmax is not None:
        assert hmax >= hmin

    inside = ef(np.array([-1500.0, 2500.0, 2500.0]))
    outside = ef(np.array([-4000.0, -500.0, 6000.0]))
    projected = ef(np.array([-3000.0, 0.0, 5000.0]))

    assert np.isfinite(inside)
    assert inside >= 150.0
    assert outside == pytest.approx(projected)


def test_define_winslow_points_3d_classifies_no_water_boundaries():
    points = np.array(
        [
            [0.0, 0.0, -8.0],   # corner -> locked
            [0.0, 0.0, -4.0],   # x/y edge -> Z only
            [0.0, 6.0, -8.0],   # x/z edge -> Y only
            [5.0, 0.0, -8.0],   # y/z edge -> X only
            [0.0, 6.0, -4.0],   # x face -> Y/Z
            [5.0, 0.0, -4.0],   # y face -> X/Z
            [5.0, 6.0, -8.0],   # bottom face -> X/Y
            [5.0, 6.0, -4.0],   # interior -> all
            [5.0, 6.0, 0.0],    # top face -> X/Y
        ]
    )

    result = define_winslow_points_3d(
        gmsh=None,
        points_3d=points,
        tag_to_index={},
        length_x=10.0,
        length_y=12.0,
        depth_z=8.0,
        padding_type=None,
        padding_x=0.0,
        padding_y=0.0,
        padding_z=0.0,
        water_interface=False,
        tol=1.0e-8,
    )

    assert result["locked"] == {0}
    assert result["move_all"] == {7}
    assert result["move_X_only"] == {3, 5, 6, 8}
    assert result["move_Y_only"] == {2, 4, 6, 8}
    assert result["move_Z_only"] == {1, 4, 5}
    assert result["water_nodes"] == set()
    assert result["movable_nodes"] == set(range(1, 9))


def test_define_winslow_points_3d_rectangular_padding_planes():
    points = np.array(
        [
            [-2.0, -3.0, -12.0],
            [-2.0, 6.0, -6.0],
            [5.0, -3.0, -6.0],
            [5.0, 6.0, -12.0],
            [5.0, 6.0, -6.0],
        ]
    )

    result = define_winslow_points_3d(
        gmsh=None,
        points_3d=points,
        tag_to_index={},
        length_x=10.0,
        length_y=12.0,
        depth_z=8.0,
        padding_type="rectangular",
        padding_x=2.0,
        padding_y=3.0,
        padding_z=4.0,
        water_interface=False,
        tol=1.0e-8,
    )

    assert 0 in result["locked"]
    assert 1 in result["move_Y_only"]
    assert 1 in result["move_Z_only"]
    assert 2 in result["move_X_only"]
    assert 2 in result["move_Z_only"]
    assert 3 in result["move_X_only"]
    assert 3 in result["move_Y_only"]
    assert result["move_all"] == {4}


def test_define_winslow_points_3d_validation():
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        define_winslow_points_3d(
            None,
            np.zeros((2, 2)),
            {},
            1.0,
            1.0,
            1.0,
            None,
            0.0,
            0.0,
            0.0,
            False,
        )

    with pytest.raises(ValueError, match="hyperelliptical"):
        define_winslow_points_3d(
            None,
            np.zeros((2, 3)),
            {},
            1.0,
            1.0,
            1.0,
            "hyperelliptical",
            0.0,
            0.0,
            0.0,
            False,
        )

    with pytest.raises(ValueError, match="tolerance must be positive"):
        define_winslow_points_3d(
            None,
            np.zeros((2, 3)),
            {},
            1.0,
            1.0,
            1.0,
            None,
            0.0,
            0.0,
            0.0,
            False,
            tol=0.0,
        )

    with pytest.raises(RuntimeError, match="No movable nodes"):
        define_winslow_points_3d(
            None,
            np.array([[0.0, 0.0, -1.0]]),
            {},
            1.0,
            1.0,
            1.0,
            None,
            0.0,
            0.0,
            0.0,
            False,
            tol=1.0e-8,
        )


@pytest.mark.slow
def test_define_winslow_points_3d_real_water_group(gmsh_session):
    gmsh = gmsh_session
    points, tag_to_index = _build_water_and_subsurface_mesh(gmsh, "Water")

    result = define_winslow_points_3d(
        gmsh=gmsh,
        points_3d=points,
        tag_to_index=tag_to_index,
        length_x=10.0,
        length_y=12.0,
        depth_z=8.0,
        padding_type=None,
        padding_x=0.0,
        padding_y=0.0,
        padding_z=0.0,
        water_interface=True,
        tol=1.0e-6,
    )

    assert result["water_nodes"]
    assert result["water_nodes"].issubset(result["locked"])
    assert result["movable_nodes"]
    assert result["water_nodes"].isdisjoint(result["movable_nodes"])


@pytest.mark.slow
def test_define_winslow_points_3d_real_water_with_padding_group(gmsh_session):
    gmsh = gmsh_session
    points, tag_to_index = _build_water_and_subsurface_mesh(
        gmsh,
        "Water_with_padding",
    )

    result = define_winslow_points_3d(
        gmsh=gmsh,
        points_3d=points,
        tag_to_index=tag_to_index,
        length_x=10.0,
        length_y=12.0,
        depth_z=8.0,
        padding_type="rectangular",
        padding_x=2.0,
        padding_y=2.0,
        padding_z=2.0,
        water_interface=True,
        tol=1.0e-6,
    )

    assert result["water_nodes"]
    assert result["water_nodes"].issubset(result["locked"])
    assert result["movable_nodes"]


def test_define_winslow_points_3d_missing_water_group(gmsh_session):
    gmsh = gmsh_session
    points = np.array([[5.0, 6.0, -4.0]])

    with pytest.raises(RuntimeError, match="Physical volume group 'Water' was not found"):
        define_winslow_points_3d(
            gmsh=gmsh,
            points_3d=points,
            tag_to_index={},
            length_x=10.0,
            length_y=12.0,
            depth_z=8.0,
            padding_type=None,
            padding_x=0.0,
            padding_y=0.0,
            padding_z=0.0,
            water_interface=True,
            tol=1.0e-6,
        )


def test_define_winslow_points_3d_empty_water_group(gmsh_session):
    gmsh = gmsh_session
    water = gmsh.model.occ.addBox(0.0, 0.0, -2.0, 10.0, 12.0, 2.0)
    gmsh.model.occ.synchronize()
    water_group = gmsh.model.addPhysicalGroup(3, [water])
    gmsh.model.setPhysicalName(3, water_group, "Water")

    with pytest.raises(RuntimeError, match="contains no mesh nodes"):
        define_winslow_points_3d(
            gmsh=gmsh,
            points_3d=np.array([[5.0, 6.0, -4.0]]),
            tag_to_index={},
            length_x=10.0,
            length_y=12.0,
            depth_z=8.0,
            padding_type=None,
            padding_x=0.0,
            padding_y=0.0,
            padding_z=0.0,
            water_interface=True,
            tol=1.0e-6,
        )


@pytest.mark.slow
def test_define_winslow_points_3d_rejects_all_water_nodes(gmsh_session):
    gmsh = gmsh_session
    water = gmsh.model.occ.addBox(0.0, 0.0, -8.0, 10.0, 12.0, 8.0)
    gmsh.model.occ.synchronize()
    water_group = gmsh.model.addPhysicalGroup(3, [water])
    gmsh.model.setPhysicalName(3, water_group, "Water")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0)
    gmsh.model.mesh.generate(3)

    points, tag_to_index = _extract_gmsh_points(gmsh)

    with pytest.raises(RuntimeError, match="Every mesh node was classified as Water"):
        define_winslow_points_3d(
            gmsh=gmsh,
            points_3d=points,
            tag_to_index=tag_to_index,
            length_x=10.0,
            length_y=12.0,
            depth_z=8.0,
            padding_type=None,
            padding_x=0.0,
            padding_y=0.0,
            padding_z=0.0,
            water_interface=True,
            tol=1.0e-6,
        )
