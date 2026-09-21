import spyro
from spyro import create_transect
import pytest
import os
import numpy as np


def is_seismicmesh_installed():
    try:
        import SeismicMesh  # noqa: F401
        return True
    except ImportError:
        return False


def get_wave_obj():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "abc_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    return spyro.examples.Camembert_acoustic(dictionary=dictionary)


def test_plot():
    rectangle_dictionary = {}
    rectangle_dictionary["mesh"] = {
        "length_z": 0.75,  # depth in km - always positive
        "length_x": 1.5,
        "h": 0.05,
    }
    rectangle_dictionary["acquisition"] = {
        "source_locations": [(-0.1, 0.75)],
        "receiver_locations": create_transect((-0.10, 0.1), (-0.10, 1.4), 50),
        "frequency": 8.0,
    }
    rectangle_dictionary["time_axis"] = {
        "final_time": 1.5,  # Final time for event
    }
    wave = spyro.examples.Rectangle_acoustic(
        dictionary=rectangle_dictionary
    )
    layer_values = [1.5, 3.0]
    z_switches = [-0.5]
    wave.multiple_layer_velocity_model(z_switches, layer_values)
    wave.forward_solve()
    spyro.plots.plot_shots(wave, show=False, filename="test_plot", file_format="png")
    expected_file = "test_plot[0].png"
    assert os.path.exists(expected_file)

    spyro.plots.debug_plot(wave.u_n, filename="test_debug_plot.png")
    expected_file = "test_debug_plot.png"
    assert os.path.exists(expected_file)

    spyro.plots.debug_pvd(wave.u_n, filename="test_debug_plot1.pvd")
    expected_file = "test_debug_plot1.pvd"
    assert os.path.exists(expected_file)


@pytest.mark.skipif(not is_seismicmesh_installed(), reason="SeismicMesh is not installed")
@pytest.mark.older_firedrake
def test_plot_mesh_sizes():
    mesh_filename = "test_mesh_for_plots.msh"
    Lz = 1.0
    Lx = 2.0
    c = 1.5
    freq = 5.0
    lbda = c/freq
    pad = 0.3
    cpw = 3

    input_mesh_parameters = {
        "length_z": Lz,
        "length_x": Lx,
        "length_y": 0.0,
        "cell_type": "triangle",
        "mesh_type": "SeismicMesh",
        "dx": None,
        "periodic": False,
        "velocity_model_file": None,
        "cells_per_wavelength": cpw,
        "source_frequency": freq,
        "minimum_velocity": c,
        "abc_pad_length": pad,
        "lbda": lbda,
        "dimension": 2,
        "edge_length": lbda/cpw,
        "output_filename": mesh_filename,
    }

    mesh_parameters = spyro.meshing.MeshingParameters()
    mesh_parameters.set_mesh(input_mesh_parameters=input_mesh_parameters)

    Mesh_obj = spyro.meshing.AutomaticMesh(
        mesh_parameters=mesh_parameters,
    )

    mesh = Mesh_obj.create_mesh()  # noqa: F841

    image_output_filename = "mesh_size.png"
    spyro.plots.plot_mesh_sizes(mesh=mesh, output_filename=image_output_filename, show=False)
    assert os.path.exists(str(image_output_filename))


@pytest.mark.newer_firedrake
def test_plot_model_in_p1():
    wave = get_wave_obj()
    filename = "model_p1.png"
    spyro.plots.plot_model_in_p1(wave, filename=str(filename), show=False)
    assert os.path.exists(str(filename))


@pytest.mark.newer_firedrake
def test_plot_model_material_parameters(tmp_path) -> None:
    """Without ``fields``, a solver's own material parameters are drawn.

    An acoustic solver has its velocity; an isotropic elastic one, whose
    model comes from the input dictionary and has not been built by a
    forward solve, its density and two wave speeds, one panel each, named
    after the parameters.
    """
    from tests.on_one_core.test_fwi_automated_adjoint import (
        ELASTIC_GUESS, build_elastic_dictionary,
    )

    figure = spyro.plots.plot_model(get_wave_obj(), tmp_path / "acoustic.png")
    assert [axis.get_title() for axis in figure.axes[:1]] == ["p_wave_velocity"]

    elastic = spyro.IsotropicWave(dictionary=build_elastic_dictionary(ELASTIC_GUESS))
    elastic.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    figure = spyro.plots.plot_model(elastic, tmp_path / "elastic.png", high_resolution=True)
    assert (tmp_path / "elastic.png").exists()
    # The panels come first in the figure, then their colour bars.
    assert [axis.get_title() for axis in figure.axes[:3]] == [
        "density", "p_wave_velocity", "s_wave_velocity",
    ]
    assert len(figure.axes) == 6


@pytest.mark.newer_firedrake
@pytest.mark.parametrize("high_resolution", [False, True])
@pytest.mark.parametrize("quadrilateral", [False, True])
def test_plot_model_fields(tmp_path, high_resolution: bool, quadrilateral: bool) -> None:
    """Plot multiple fields through the shared model plotting entry point.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    high_resolution : bool
        Whether to regrid onto a finer CG1 mesh.
    quadrilateral : bool
        Whether the original mesh uses quadrilaterals.

    Returns
    -------
    None
        Assertions verify geometry, values, panel layout and unchanged inputs.
    """
    import firedrake as fire
    from types import SimpleNamespace

    mesh = fire.RectangleMesh(4, 4, 1.0, 2.0, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    space = fire.FunctionSpace(mesh, "CG", 2)
    z, x = fire.SpatialCoordinate(mesh)
    velocity = fire.Function(space).interpolate(1.5 + z * z)
    gradient = fire.Function(space).interpolate(x - 1.0)
    wave = SimpleNamespace(
        initial_velocity_model=velocity,
        source_locations=[(-0.1, 1.0)],
        receiver_locations=create_transect((-0.9, 0.5), (-0.9, 1.5), 5),
        mesh_parameters=spyro.meshing.MeshingParameters(
            input_mesh_dictionary={"length_z": 1.0, "length_x": 2.0,
                                   "length_y": 0.0, "dimension": 2},
            comm=fire.Ensemble(mesh.comm, mesh.comm.size),
        ),
    )
    coordinates = mesh.coordinates.dat.data_ro.copy()
    values = velocity.dat.data_ro.copy()
    figure = spyro.plots.plot_model(
        wave, tmp_path / "fields.png", fields=[velocity, gradient],
        titles=["velocity", "gradient"], vmin=[1.5, -1.0], vmax=[2.5, 1.0],
        high_resolution=high_resolution, high_resolution_grid_value=0.13,
    )
    assert (tmp_path / "fields.png").exists()
    assert len(figure.axes) == 4
    assert figure.axes[0].get_xlabel() == "x (km)"
    assert figure.axes[0].get_ylabel() == "z (km)"
    assert figure.axes[0].collections[0].get_clim() == (1.5, 2.5)
    assert np.isclose(figure.axes[0].collections[0].get_array().min(), 1.5)
    assert np.isclose(figure.axes[0].collections[0].get_array().max(), 2.5)
    assert np.array_equal(coordinates, mesh.coordinates.dat.data_ro)
    assert np.array_equal(values, velocity.dat.data_ro)
    figure = spyro.plots.plot_model(
        wave, tmp_path / "grid.png", fields=[velocity, gradient, velocity],
        columns=2, high_resolution=high_resolution, high_resolution_grid_value=0.13,
        show_acquisition=False,
    )
    assert len([axis for axis in figure.axes if axis.get_visible()]) == 6
    assert len(figure.axes) == 7
    assert len(figure.axes[0].collections) == 1

    # Existing single-model calls still work, including the non-flipped axes.
    figure = spyro.plots.plot_model(
        wave, tmp_path / "single.png", high_resolution=high_resolution,
        high_resolution_grid_value=0.13, flip_axis=False,
        abc_points=[(-0.2, 0.2), (-0.8, 0.2), (-0.8, 1.8)],
    )
    assert figure.axes[0].get_xlabel() == "z (km)"
    assert len(figure.axes[0].lines[0].get_xdata()) == 4
    for invalid in ([], [fire.Function(fire.VectorFunctionSpace(mesh, "CG", 1))]):
        with pytest.raises(ValueError):
            spyro.plots.plot_model(wave, fields=invalid)
    with pytest.raises(ValueError, match="titles"):
        spyro.plots.plot_model(wave, fields=[velocity, gradient], titles=["one"])
    with pytest.raises(ValueError, match="columns"):
        spyro.plots.plot_model(wave, columns=0)
    with pytest.raises(ValueError, match="no material model"):
        spyro.plots.plot_model(SimpleNamespace(initial_velocity_model=None))


@pytest.mark.parametrize("spacing", [0.0, -0.1, np.nan, np.inf])
def test_model_plot_rejects_invalid_spacing(spacing: float) -> None:
    """Reject invalid regridding distances before accessing the mesh.

    Parameters
    ----------
    spacing : float
        Invalid sampling distance.

    Returns
    -------
    None
        An assertion checks the validation error.
    """
    with pytest.raises(ValueError, match="grid_spacing must be finite and positive"):
        spyro.utils.change_scalar_field_resolution(None, None, spacing)


def test_plot_receiver_response(tmp_path):
    receiver_data = np.sin(np.linspace(0.0, 2.0 * np.pi, 100))
    output_file = tmp_path / "receiver_response.png"

    spyro.plots.plot_receiver_response(
        receiver_data,
        final_time=2.0,
        show=False,
        filename=str(output_file),
        receiver_id_for_title=7,
        name="trace",
    )

    assert output_file.exists()


if __name__ == "__main__":
    test_plot()
    test_plot_mesh_sizes()
    test_plot_model_in_p1()
