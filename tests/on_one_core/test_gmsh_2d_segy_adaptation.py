"""Wave-adapted 2D Gmsh mesh built from a SEG-Y velocity model.

The SEG-Y follows the layout used in ``notebook_tutorials/meshing_2d.ipynb``:
each trace is a depth profile at one x position, and sample 0 is the surface.
The model has three regions with different velocities, laid out so that a
z/x swap, a flipped depth axis or a mirrored x axis would put the wrong element
size in at least one region:

    depth < 1 km                 : 1.5 km/s  ->  h = 0.075 km
    depth > 1 km, x < 1.5 km     : 2.5 km/s  ->  h = 0.125 km
    depth > 1 km, x > 1.5 km     : 4.0 km/s  ->  h = 0.200 km

with h = vp / (frequency * cells_per_wavelength). Lengths are in km and
velocities in km/s, following the spyro convention.
"""

import firedrake as fire
import numpy as np
import pytest
import segyio

from spyro.meshing import AutomaticMesh, MeshingParameters


LENGTH_Z = 2.0
LENGTH_X = 3.0
GRID_SPACING = 0.05
INTERFACE_DEPTH = 1.0
INTERFACE_X = 1.5
VP_TOP = 1.5
VP_BOTTOM_LEFT = 2.5
VP_BOTTOM_RIGHT = 4.0
FREQUENCY = 10.0
CELLS_PER_WAVELENGTH = 2.0
# Cells closer than this to a velocity interface are in the size transition
# and are left out of the size check.
INTERFACE_MARGIN = 0.25
SIZE_TOLERANCE = 0.1


def _write_segy(filename) -> None:
    """Write the three-region velocity model as a standard SEG-Y file.

    Parameters
    ----------
    filename : pathlib.Path
        Output SEG-Y path.
    """
    nz = int(round(LENGTH_Z / GRID_SPACING)) + 1
    nx = int(round(LENGTH_X / GRID_SPACING)) + 1
    depth = np.arange(nz) * GRID_SPACING
    x = np.arange(nx) * GRID_SPACING
    vp = np.where(
        depth[:, None] < INTERFACE_DEPTH,
        VP_TOP,
        np.where(x[None, :] < INTERFACE_X, VP_BOTTOM_LEFT, VP_BOTTOM_RIGHT),
    ).astype(np.float32)

    spec = segyio.spec()
    spec.format = 1
    spec.samples = range(nz)
    spec.tracecount = nx
    with segyio.create(str(filename), spec) as f:
        for i in range(nx):
            f.trace[i] = vp[:, i]


def _cell_centroids_and_sizes(mesh: fire.MeshGeometry) -> tuple:
    """Compute the centroid and mean edge length of every triangle.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Triangular mesh with linear coordinates.

    Returns
    -------
    centroids : numpy.ndarray
        Array of shape ``(num_cells, 2)`` with ``(z, x)`` centroids.
    sizes : numpy.ndarray
        Array of shape ``(num_cells,)`` with the mean edge length per cell.
    """
    vertices = mesh.coordinates.dat.data_ro[mesh.coordinates.cell_node_map().values]
    edges = vertices - np.roll(vertices, 1, axis=1)
    return vertices.mean(axis=1), np.linalg.norm(edges, axis=2).mean(axis=1)


@pytest.fixture
def adapted_mesh(tmp_path) -> fire.MeshGeometry:
    """Mesh the three-region SEG-Y model with Gmsh through ``AutomaticMesh``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory for the SEG-Y and mesh files.

    Returns
    -------
    firedrake.MeshGeometry
        The initialised Firedrake mesh.
    """
    segy_file = tmp_path / "three_regions.segy"
    _write_segy(segy_file)
    parameters = {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "output_filename": str(tmp_path / "three_regions.msh"),
        "segy_velocity_model": str(segy_file),
        "cells_per_wavelength": CELLS_PER_WAVELENGTH,
        "source_frequency": FREQUENCY,
        # No smoothing, so the target size inside each region is exact.
        "grade": 0.0,
        "hmin_segy": 0.0,
        "padding_type": None,
        "water_interface": False,
        "structured_mesh": False,
        "apply_winslow": False,
    }
    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters)
    mesh = AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()
    mesh.init()
    return mesh


def test_gmsh_2d_segy_domain_uses_spyro_axes(adapted_mesh) -> None:
    """Check that the mesh covers ``z in [-Lz, 0]`` and ``x in [0, Lx]``.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh built from the three-region SEG-Y model.
    """
    coordinates = adapted_mesh.coordinates.dat.data_ro
    np.testing.assert_allclose(coordinates.min(axis=0), [-LENGTH_Z, 0.0], atol=1e-8)
    np.testing.assert_allclose(coordinates.max(axis=0), [0.0, LENGTH_X], atol=1e-8)


@pytest.mark.parametrize(
    "region, vp",
    [
        ("top", VP_TOP),
        ("bottom_left", VP_BOTTOM_LEFT),
        ("bottom_right", VP_BOTTOM_RIGHT),
    ],
)
def test_gmsh_2d_segy_element_size(adapted_mesh, region: str, vp: float) -> None:
    """Check that element sizes follow ``vp / (frequency * cpw)`` in each region.

    Parameters
    ----------
    adapted_mesh : firedrake.MeshGeometry
        Mesh built from the three-region SEG-Y model.
    region : str
        Name of the velocity region to check.
    vp : float
        P-wave velocity of that region.
    """
    centroids, sizes = _cell_centroids_and_sizes(adapted_mesh)
    z, x = centroids[:, 0], centroids[:, 1]

    shallow = z > -INTERFACE_DEPTH + INTERFACE_MARGIN
    deep = z < -INTERFACE_DEPTH - INTERFACE_MARGIN
    region_mask = {
        "top": shallow,
        "bottom_left": deep & (x < INTERFACE_X - INTERFACE_MARGIN),
        "bottom_right": deep & (x > INTERFACE_X + INTERFACE_MARGIN),
    }[region]
    assert region_mask.sum() >= 20, f"Too few cells in region {region}."

    expected_size = vp / (FREQUENCY * CELLS_PER_WAVELENGTH)
    median_size = np.median(sizes[region_mask])
    assert np.isclose(median_size, expected_size, rtol=SIZE_TOLERANCE), (
        f"Region {region}: median element size {median_size:.3f} km, "
        f"expected {expected_size:.3f} km."
    )
