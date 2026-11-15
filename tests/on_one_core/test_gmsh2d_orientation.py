"""Check that gmsh meshes keep the model where the velocity readers put it."""

from collections.abc import Callable
from pathlib import Path

import firedrake as fire
import numpy as np
import pytest
import ufl

import spyro
from spyro.io import write_function_to_grid
from spyro.io.segy_io import create_segy_from_grid

LENGTH_Z = 1.0
LENGTH_X = 2.0
GRID_SPACING = 0.05
PADDING = 0.3


def seabed(x: ufl.core.expr.Expr | np.ndarray) -> ufl.core.expr.Expr | np.ndarray:
    """Return a seabed that is 0.4 km deeper on the right than on the left.

    Parameters
    ----------
    x : `ufl.core.expr.Expr` or `numpy.ndarray`
        Lateral coordinate, in km.

    Returns
    -------
    `ufl.core.expr.Expr` or `numpy.ndarray`
        Seabed `z` coordinate, in km.
    """
    return -0.3 - 0.2 * x


def shear_velocity(z: ufl.core.expr.Expr, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Return a Vs that is zero in the water above `seabed`.

    Parameters
    ----------
    z : `ufl.core.expr.Expr`
        Depth coordinate, in km.
    x : `ufl.core.expr.Expr`
        Lateral coordinate, in km.

    Returns
    -------
    `ufl.core.expr.Expr`
        Vs, in km/s.
    """
    return fire.conditional(z < seabed(x), 1.5 - 0.5 * z, 0.0)


def segy_model(directory: Path) -> dict:
    """Write the Vs model to a SEG-Y file whose samples are mesh nodes.

    Parameters
    ----------
    directory : `pathlib.Path`
        Directory of the SEG-Y file.

    Returns
    -------
    `dict`
        Mesh parameters that give the SEG-Y file to the mesher.
    """
    nz = round(LENGTH_Z / GRID_SPACING)
    nx = round(LENGTH_X / GRID_SPACING)
    # Lx and Ly are the coordinates of the upper corner, not lengths
    mesh = fire.RectangleMesh(nz, nx, 0.0, LENGTH_X, originX=-LENGTH_Z)
    V = fire.FunctionSpace(mesh, "CG", 1)
    vs = fire.Function(V).interpolate(shear_velocity(*fire.SpatialCoordinate(mesh)))
    filename = str(directory / "vs.segy")
    create_segy_from_grid(write_function_to_grid(vs, V, GRID_SPACING), filename)
    return {"segy_velocity_model": filename}


def grid_model(directory: Path) -> dict:
    """Sample the Vs model on a grid dictionary.

    Parameters
    ----------
    directory : `pathlib.Path`
        Unused, present to share the signature of `segy_model`.

    Returns
    -------
    `dict`
        Mesh parameters that give the grid dictionary to the mesher.
    """
    mesh = spyro.meshing.AutomaticMesh(
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary={
                "dimension": 2,
                "length_z": LENGTH_Z,
                "length_x": LENGTH_X,
                "length_y": 0.0,
                "mesh_type": "firedrake_mesh",
                "edge_length": GRID_SPACING,
            }
        )
    ).create_mesh()
    grid = spyro.utils.scalar_conditional_to_grid(
        conditional=shear_velocity(*fire.SpatialCoordinate(mesh)),
        domain_dimensions=(LENGTH_Z, LENGTH_X),
        grid_spacing=GRID_SPACING,
    )
    return {"velocity_model": grid}


@pytest.mark.parametrize("model", [segy_model, grid_model])
@pytest.mark.parametrize("padding_type", [None, "rectangular"])
def test_gmsh_water_interface_follows_model(
    model: Callable[[Path], dict],
    padding_type: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that the marked seabed is not mirrored or shifted in x.

    Parameters
    ----------
    model : callable
        `segy_model` or `grid_model`.
    padding_type : `str` or `None`
        Padding of the gmsh mesh.
    tmp_path : `pathlib.Path`
        Pytest temporary directory.
    monkeypatch : `pytest.MonkeyPatch`
        Pytest fixture, used to write the mesher files in `tmp_path`.

    Returns
    -------
    None
    """
    pytest.importorskip("gmsh")
    monkeypatch.chdir(tmp_path)
    mesh_parameters = spyro.meshing.MeshingParameters(
        input_mesh_dictionary={
            "mesh_type": "gmsh_mesh",
            "dimension": 2,
            "length_z": LENGTH_Z,
            "length_x": LENGTH_X,
            "output_filename": str(tmp_path / "mesh.msh"),
            "cells_per_wavelength": 2.0,
            "source_frequency": 5.0,
            "padding_type": padding_type,
            "padding_x": PADDING,
            "padding_z": PADDING,
            "h_padding": PADDING,
            "grade": None,
            "water_interface": True,
            "water_search_value": 0.0,
            "vp_water": 1.5,
            "structured_mesh": False,
            "apply_winslow": False,
            **model(tmp_path),
        }
    )
    mesh = spyro.meshing.AutomaticMesh(mesh_parameters).create_mesh()

    coords = mesh.coordinates.dat.data_ro
    pad = 0.0 if padding_type is None else PADDING
    assert np.allclose(coords.min(axis=0), [-LENGTH_Z - pad, -pad])
    assert np.allclose(coords.max(axis=0), [0.0, LENGTH_X + pad])

    # Gmsh physical groups: WaterSurface cells are marked 1, SubSurface 2
    cell_vertices = fire.FunctionSpace(mesh, "CG", 1).cell_node_map().values
    water = np.zeros(len(coords), dtype=bool)
    rock = np.zeros(len(coords), dtype=bool)
    water[cell_vertices[mesh.cell_subset(1).indices]] = True
    rock[cell_vertices[mesh.cell_subset(2).indices]] = True
    z, x = coords[water & rock].T
    assert x.min() == pytest.approx(0.0)
    assert x.max() == pytest.approx(LENGTH_X)
    # The mesher puts the seabed at the first rock sample below it. A mirrored
    # mesh would be up to 0.4 km away at the sides.
    assert np.all(np.abs(z - seabed(x)) < 1.2 * GRID_SPACING)
