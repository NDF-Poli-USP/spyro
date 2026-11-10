"""Boundary tags of homogeneous 2D Gmsh meshes, as seen by Firedrake.

The mesh is deliberately coarse (about three elements per side) so the test
runs in about a second. Lengths are in km and velocities in km/s.
"""

import firedrake as fire
import numpy as np

from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters


VP = 1.5
VS = 0.8
LENGTH_Z = 1.0
LENGTH_X = 1.2
GRID_SPACING = 0.1
TOLERANCE = 1e-6

# Expected boundary tags: id -> (name, coordinate index, coordinate value).
# Coordinates follow the spyro convention (z, x), with z <= 0. The ids match
# Firedrake's RectangleMesh, which the local absorbing boundaries rely on.
BOUNDARY_TAGS = {
    1: ("Bottom", 0, -LENGTH_Z),
    2: ("Top", 0, 0.0),
    3: ("Left", 1, 0.0),
    4: ("Right", 1, LENGTH_X),
}


def _homogeneous_mesh_2d(tmp_path) -> fire.MeshGeometry:
    """Build a coarse 2D Gmsh mesh with homogeneous vp and vs and no padding.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory where the mesh file is written.

    Returns
    -------
    firedrake.MeshGeometry
        The initialised Firedrake mesh.
    """
    shape = (
        int(round(LENGTH_Z / GRID_SPACING)) + 1,
        int(round(LENGTH_X / GRID_SPACING)) + 1,
    )
    parameters = {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "output_filename": str(tmp_path / "homogeneous_2d.msh"),
        "velocity_model": {
            "vp_values": np.full(shape, VP),
            "vs_values": np.full(shape, VS),
        },
        "cells_per_wavelength": 2.0,
        "source_frequency": 2.0,
        "padding_type": None,
        "water_interface": False,
        "structured_mesh": False,
        "apply_winslow": False,
    }

    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters)
    mesh = AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()
    mesh.init()
    return mesh


def test_gmsh_2d_homogeneous_boundary_tags(tmp_path, monkeypatch) -> None:
    """Check that every outer boundary is tagged and reachable through ``ds``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to run in ``tmp_path``, since the mesher writes a temporary
        velocity file to the current working directory.
    """
    monkeypatch.chdir(tmp_path)
    mesh = _homogeneous_mesh_2d(tmp_path)
    coords = fire.SpatialCoordinate(mesh)
    lengths = np.array([LENGTH_Z, LENGTH_X])

    markers = set(int(m) for m in mesh.exterior_facets.unique_markers)
    assert markers == set(BOUNDARY_TAGS), (
        f"Expected boundary ids {sorted(BOUNDARY_TAGS)}, got {sorted(markers)}."
    )

    tagged_length = 0.0
    for tag, (name, axis, value) in BOUNDARY_TAGS.items():
        ds_tag = fire.ds(tag, domain=mesh)

        length = fire.assemble(fire.Constant(1.0) * ds_tag)
        expected_length = lengths[1 - axis]
        assert np.isclose(length, expected_length, rtol=TOLERANCE), (
            f"{name} (id {tag}) has length {length}, expected {expected_length}."
        )

        # Every facet carrying this tag must lie on the expected line.
        offset = fire.assemble((coords[axis] - value) ** 2 * ds_tag)
        assert offset < TOLERANCE * expected_length, (
            f"{name} (id {tag}) is not on coordinate {axis} == {value}."
        )
        tagged_length += length

    total_length = fire.assemble(fire.Constant(1.0) * fire.ds(domain=mesh))
    assert np.isclose(tagged_length, total_length, rtol=TOLERANCE), (
        "Some exterior facets are not tagged."
    )
