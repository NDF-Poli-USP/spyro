"""3D Gmsh meshes of a small synthetic binary velocity model.

The model is small enough for each mesh to take a few seconds. The box
lengths differ along every axis and the velocity layout is asymmetric along
every axis, so swapping or mirroring any axis puts the wrong element size in at
least one region:

    z > -500 m (upper half), y < 1000 m : 1500 m/s  ->  h = 150 m
    z > -500 m (upper half), y > 1000 m : 2000 m/s  ->  h = 200 m
    z < -500 m (lower half), x <  750 m : 2500 m/s  ->  h = 250 m
    z < -500 m (lower half), x >  750 m : 4000 m/s  ->  h = 400 m

with h = vp / (frequency * cells_per_wavelength). The arrays use the spyro
`(z, x, y)` axis order, where index 0 along z is the surface.

Lengths are in metres, unlike most spyro tests: the 3D Gmsh sizing assumes
metres (SeismicMesh defaults to `hmin = 150` m and checks that `vp_water`
lies between 1300 and 1800 m/s).

The SeismicMesh sizing path is switched off in these tests, because the size
function it returns cannot be evaluated and Gmsh then never finishes meshing.
`test_create_sizing_function3d_seismicmesh_path` in
`test_meshing_utils_3d.py` tracks that bug.
"""

import firedrake as fire
import numpy as np
import pytest

import spyro.meshing.meshing_utils3d

from .gmsh_mesh_checks import (
    assert_bounds,
    assert_cell_regions,
    assert_cell_type,
    assert_measure,
    assert_no_degenerate_cells,
    assert_region_above,
    create_gmsh_mesh,
    measure,
    physical_groups,
)


LENGTH_Z = 1000.0
LENGTH_X = 1500.0
LENGTH_Y = 2000.0
SPACING = 50.0
LAYER_DEPTH = 500.0
SPLIT_X = 750.0
SPLIT_Y = 1000.0
VP_UPPER_NEAR = 1500.0
VP_UPPER_FAR = 2000.0
VP_LOWER_LEFT = 2500.0
VP_LOWER_RIGHT = 4000.0
FREQUENCY = 5.0
CELLS_PER_WAVELENGTH = 2.0
# Cells closer than this to a velocity interface are in the size transition
# and are left out of the size check.
INTERFACE_MARGIN = 200.0
# Gmsh tetrahedra have a mean edge length somewhat above the target size,
# more so in regions only a few elements thick.
SIZE_RATIO_RANGE = (0.9, 1.4)
PADDING = 300.0
WATER_DEPTH = 200.0
VP_WATER = 1500.0


def _grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the depth, x and y sample positions of the velocity grid.

    Returns
    -------
    depth, x, y : numpy.ndarray
        Sample positions along each axis, starting at zero.
    """
    shape = [int(round(length / SPACING)) + 1 for length in (LENGTH_Z, LENGTH_X, LENGTH_Y)]
    return tuple(np.arange(n) * SPACING for n in shape)


def _velocity_cube(water: bool = False) -> np.ndarray:
    """Build the four-region velocity cube in `(z, x, y)` order.

    Parameters
    ----------
    water : bool, optional
        If True, set the top `WATER_DEPTH` metres to zero velocity, which the
        mesher detects as water.

    Returns
    -------
    numpy.ndarray
        Velocity values of shape `(nz, nx, ny)`.
    """
    depth, x, y = _grid_axes()
    Z, X, Y = np.meshgrid(depth, x, y, indexing="ij")
    vp = np.where(
        Z < LAYER_DEPTH,
        np.where(Y < SPLIT_Y, VP_UPPER_NEAR, VP_UPPER_FAR),
        np.where(X < SPLIT_X, VP_LOWER_LEFT, VP_LOWER_RIGHT),
    )
    if water:
        vp = np.where(Z < WATER_DEPTH - 1e-6, 0.0, vp)
    return vp


def _write_binary(filename, vp: np.ndarray) -> None:
    """Write a velocity cube as big-endian float32 in Fortran order.

    Parameters
    ----------
    filename : pathlib.Path
        Output file.
    vp : numpy.ndarray
        Velocity cube in `(z, x, y)` order.
    """
    vp.astype(">f4").ravel(order="F").tofile(filename)


def _parameters(tmp_path, velocity_file, **overrides) -> dict:
    """Mesh parameters for the synthetic model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the mesh file.
    velocity_file : pathlib.Path
        Binary velocity model.
    **overrides
        Entries that replace the defaults.

    Returns
    -------
    dict
        Input mesh dictionary for `MeshingParameters`.
    """
    nz, nx, ny = (len(axis) for axis in _grid_axes())
    parameters = {
        "mesh_type": "gmsh_mesh",
        "dimension": 3,
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "length_y": LENGTH_Y,
        "output_filename": str(tmp_path / "synthetic.msh"),
        "segy_velocity_model": str(velocity_file),
        "segy_nz": nz,
        "segy_nx": nx,
        "segy_ny": ny,
        "segy_dz": SPACING,
        "segy_dx": SPACING,
        "segy_dy": SPACING,
        "cells_per_wavelength": CELLS_PER_WAVELENGTH,
        "source_frequency": FREQUENCY,
        # No smoothing, so the target size inside each region is exact.
        "grade": 0.0,
        "hmin_segy": 0.0,
        "padding_type": None,
        "padding_x": PADDING,
        "padding_y": PADDING,
        "padding_z": PADDING,
        "h_padding": VP_LOWER_RIGHT / (FREQUENCY * CELLS_PER_WAVELENGTH),
        "water_interface": False,
        "water_search_value": 0.0,
        "vp_water": VP_WATER,
        "structured_mesh": False,
        "apply_winslow": False,
        "extend_segy": True,
    }
    parameters.update(overrides)
    return parameters


def _cell_centroids_and_sizes(mesh: fire.MeshGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Compute the centroid and mean edge length of every tetrahedron.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Tetrahedral mesh with linear coordinates.

    Returns
    -------
    centroids : numpy.ndarray
        Array of shape `(num_cells, 3)` with `(z, x, y)` centroids.
    sizes : numpy.ndarray
        Array of shape `(num_cells,)` with the mean edge length per cell.
    """
    vertices = mesh.coordinates.dat.data_ro[mesh.coordinates.cell_node_map().values]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.stack([vertices[:, i] - vertices[:, j] for i, j in pairs], axis=1)
    return vertices.mean(axis=1), np.linalg.norm(edges, axis=2).mean(axis=1)


@pytest.fixture(scope="module")
def adapted_mesh(tmp_path_factory) -> fire.MeshGeometry:
    """Mesh the four-region model without padding or water.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for the module's temporary directory.

    Returns
    -------
    firedrake.MeshGeometry
        The generated tetrahedral mesh.
    """
    tmp_path = tmp_path_factory.mktemp("synthetic_3d")
    velocity_file = tmp_path / "four_regions.bin"
    _write_binary(velocity_file, _velocity_cube())
    return _mesh_without_seismicmesh(_parameters(tmp_path, velocity_file))


def test_gmsh_3d_synthetic_domain_uses_spyro_axes(adapted_mesh) -> None:
    """Check that the mesh covers `z in [-Lz, 0]`, `x in [0, Lx]`, `y in [0, Ly]`.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh of the four-region model.
    """
    assert_cell_type(adapted_mesh, "tetrahedron")
    assert_bounds(
        adapted_mesh, (-LENGTH_Z, 0.0, 0.0), (0.0, LENGTH_X, LENGTH_Y), rtol=1e-9
    )
    assert_measure(adapted_mesh, LENGTH_Z * LENGTH_X * LENGTH_Y)
    assert_no_degenerate_cells(adapted_mesh)


REGIONS = {
    # name: velocity, in increasing order
    "upper_near": VP_UPPER_NEAR,
    "upper_far": VP_UPPER_FAR,
    "lower_left": VP_LOWER_LEFT,
    "lower_right": VP_LOWER_RIGHT,
}


def _median_size_per_region(mesh: fire.MeshGeometry) -> dict[str, float]:
    """Median element size in each velocity region, away from interfaces.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh of the four-region model.

    Returns
    -------
    dict of str to float
        Median mean-edge length of the cells in each region.
    """
    centroids, sizes = _cell_centroids_and_sizes(mesh)
    z, x, y = centroids.T
    depth = -z
    upper = depth < LAYER_DEPTH - INTERFACE_MARGIN
    lower = depth > LAYER_DEPTH + INTERFACE_MARGIN
    masks = {
        "upper_near": upper & (y < SPLIT_Y - INTERFACE_MARGIN),
        "upper_far": upper & (y > SPLIT_Y + INTERFACE_MARGIN),
        "lower_left": lower & (x < SPLIT_X - INTERFACE_MARGIN),
        "lower_right": lower & (x > SPLIT_X + INTERFACE_MARGIN),
    }
    for name, mask in masks.items():
        assert np.count_nonzero(mask) >= 20, f"Too few cells in {name}."
    return {name: float(np.median(sizes[mask])) for name, mask in masks.items()}


def test_gmsh_3d_synthetic_size_follows_velocity_order(adapted_mesh) -> None:
    """Check that slower regions get smaller elements.

    The four regions have distinct velocities, so mirroring or swapping any
    axis would break the order.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh of the four-region model.
    """
    medians = _median_size_per_region(adapted_mesh)
    ordered = [medians[name] for name in REGIONS]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered), (
        f"Element sizes {medians} do not increase with velocity."
    )


@pytest.mark.parametrize("region", list(REGIONS))
def test_gmsh_3d_synthetic_element_size(adapted_mesh, region: str) -> None:
    """Check that element sizes follow `vp / (frequency * cpw)` in each region.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh of the four-region model.
    region : str
        Name of the velocity region to check.
    """
    target = REGIONS[region] / (FREQUENCY * CELLS_PER_WAVELENGTH)
    ratio = _median_size_per_region(adapted_mesh)[region] / target
    low, high = SIZE_RATIO_RANGE
    assert low <= ratio <= high, (
        f"Median element size in {region} is {ratio:.3f} times the target "
        f"{target}, outside [{low}, {high}]."
    )


def _mesh_without_seismicmesh(parameters: dict) -> fire.MeshGeometry:
    """Build a mesh with the SeismicMesh sizing path switched off.

    Parameters
    ----------
    parameters : dict
        Input mesh dictionary.

    Returns
    -------
    firedrake.MeshGeometry
        The generated mesh.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spyro.meshing.meshing_utils3d, "SeismicMesh", None)
        return create_gmsh_mesh(parameters)


@pytest.fixture(
    scope="module",
    params=[
        None,
        "rectangular",
        pytest.param("hyperelliptical", marks=pytest.mark.slow),
    ],
)
def water_mesh(request, tmp_path_factory) -> tuple:
    """Mesh the model with a flat water layer, for each padding type.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Carries the padding type in `request.param`.
    tmp_path_factory : pytest.TempPathFactory
        Factory for the temporary directory.

    Returns
    -------
    tuple
        The padding type, the mesh and the path of the `.msh` file.
    """
    padding_type = request.param
    tmp_path = tmp_path_factory.mktemp(f"water_{padding_type}")
    velocity_file = tmp_path / "water.bin"
    _write_binary(velocity_file, _velocity_cube(water=True))
    parameters = _parameters(
        tmp_path,
        velocity_file,
        water_interface=True,
        padding_type=padding_type,
    )
    mesh = _mesh_without_seismicmesh(parameters)
    return padding_type, mesh, parameters["output_filename"]


def test_gmsh_3d_synthetic_water_regions(water_mesh, request) -> None:
    """Check the water, subsurface and padding volumes.

    The water layer is flat, so its volume is known exactly.

    Parameters
    ----------
    water_mesh : tuple
        Padding type, mesh and `.msh` path from the `water_mesh` fixture.
    request : pytest.FixtureRequest
        Used to mark the known hyperelliptical failure.
    """
    padding_type, mesh, msh_file = water_mesh
    if padding_type == "hyperelliptical":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "Known bug: with water and hyperelliptical padding, the "
                    "volume physical groups are picked by position after the "
                    "OCC fragment, and Subsurface and Padding come out swapped."
                ),
            )
        )
    box_volume = LENGTH_Z * LENGTH_X * LENGTH_Y
    water_volume = WATER_DEPTH * LENGTH_X * LENGTH_Y

    assert_cell_type(mesh, "tetrahedron")
    assert_no_degenerate_cells(mesh)
    if padding_type is None:
        assert_bounds(mesh, (-LENGTH_Z, 0.0, 0.0), (0.0, LENGTH_X, LENGTH_Y))
        assert_measure(mesh, box_volume)
    else:
        bounds_rtol = 1e-9 if padding_type == "rectangular" else 1e-2
        assert_bounds(
            mesh,
            (-LENGTH_Z - PADDING, -PADDING, -PADDING),
            (0.0, LENGTH_X + PADDING, LENGTH_Y + PADDING),
            rtol=bounds_rtol,
        )
        if padding_type == "rectangular":
            assert_measure(
                mesh,
                (LENGTH_Z + PADDING) * (LENGTH_X + 2 * PADDING)
                * (LENGTH_Y + 2 * PADDING),
            )

    cell_groups = physical_groups(msh_file)[3]
    expected_measures = {
        ("Water",): water_volume,
        ("Subsurface",): box_volume - water_volume,
    }
    if padding_type is not None:
        expected_measures[("Padding",)] = measure(mesh) - box_volume
    # Fragmenting the water and subsurface with the hyperellipsoid moves the
    # flat interface very slightly (about 1e-4 of the water volume).
    region_rtol = 1e-3 if padding_type == "hyperelliptical" else 1e-9
    assert_cell_regions(mesh, cell_groups, expected_measures, rtol=region_rtol)
    assert_region_above(mesh, cell_groups["Water"], cell_groups["Subsurface"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known bug: with rectangular padding, Water_with_padding and "
        "Subsurface_with_padding overlap Water and Subsurface. Each cell keeps "
        "only one tag, so these two groups never reach Firedrake."
    ),
)
def test_gmsh_3d_synthetic_every_cell_group_reaches_firedrake(water_mesh) -> None:
    """Check that every cell physical group is a usable Firedrake marker.

    Parameters
    ----------
    water_mesh : tuple
        Padding type, mesh and `.msh` path from the `water_mesh` fixture.
    """
    padding_type, mesh, msh_file = water_mesh
    if padding_type != "rectangular":
        pytest.skip("Only rectangular padding defines overlapping groups.")
    for name, tag in physical_groups(msh_file)[3].items():
        try:
            volume = measure(mesh, tag)
        except ValueError:
            volume = 0.0
        assert volume > 0.0, f"Cell group {name!r} is empty in Firedrake."


@pytest.mark.parametrize(
    "axes_order",
    [
        (0, 1, 2),
        pytest.param(
            (1, 0, 2),
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Known bug: a grid velocity_model is written to a temporary "
                    "file without applying segy_axes_order, then read back with "
                    "it, which scrambles the model."
                ),
            ),
        ),
    ],
    ids=["canonical_axes", "permuted_axes"],
)
def test_gmsh_3d_grid_velocity_model_matches_binary_file(
    adapted_mesh, tmp_path, monkeypatch, axes_order
) -> None:
    """Check that a grid `velocity_model` gives the same mesh as the binary.

    The grid is always in `(z, x, y)` order, so `segy_axes_order` must not
    change the result.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh of the same model read from a binary file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to run in `tmp_path`, since the mesher writes a temporary
        velocity file to the current working directory.
    axes_order : tuple of int
        Value of `segy_axes_order`.
    """
    monkeypatch.chdir(tmp_path)
    parameters = _parameters(
        tmp_path,
        velocity_file=None,
        segy_velocity_model=None,
        velocity_model={"vp_values": _velocity_cube()},
        segy_axes_order=axes_order,
    )
    mesh = _mesh_without_seismicmesh(parameters)

    assert mesh.num_cells() == adapted_mesh.num_cells()
    np.testing.assert_allclose(
        np.sort(mesh.coordinates.dat.data_ro, axis=0),
        np.sort(adapted_mesh.coordinates.dat.data_ro, axis=0),
    )
