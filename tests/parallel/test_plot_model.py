"""Check collective model plotting on a spatially distributed mesh."""
from pathlib import Path
from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest

import spyro


@pytest.mark.parallel(2)
@pytest.mark.parametrize("high_resolution", [False, True])
def test_plot_model_spatial(tmp_path: Path, high_resolution: bool) -> None:
    """Gather a complete field and save on spatial rank zero.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory shared from spatial rank zero.
    high_resolution : bool
        Whether to use CG1 regridding before drawing.

    Returns
    -------
    None
        Assertions verify rank ownership, field values and domain coverage.
    """
    mesh = fire.RectangleMesh(4, 4, 1.0, 2.0, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] *= -1
    space = fire.FunctionSpace(mesh, "CG", 2)
    z, x = fire.SpatialCoordinate(mesh)
    field = fire.Function(space).interpolate(z * z + x)
    wave = SimpleNamespace(
        initial_velocity_model=field, source_locations=[], receiver_locations=[],
        mesh_parameters=spyro.meshing.MeshingParameters(
            input_mesh_dictionary={"length_z": 1.0, "length_x": 2.0,
                                   "length_y": 0.0, "dimension": 2},
            comm=fire.Ensemble(mesh.comm, mesh.comm.size),
        ),
    )
    output = Path(mesh.comm.bcast(str(tmp_path), root=0)) / "model.png"
    figure = spyro.plots.plot_model(
        wave, output, high_resolution=high_resolution, high_resolution_grid_value=0.1,
    )
    if mesh.comm.rank == 0:
        assert output.exists()
        axis = figure.axes[0]
        assert np.allclose(axis.dataLim.bounds, [0.0, -1.0, 2.0, 1.0])
        values = axis.collections[0].get_array()
        assert np.isclose(values.min(), 0.0)
        assert np.isclose(values.max(), 3.0)
    else:
        assert figure is None
