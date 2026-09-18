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
def test_plot_model_in_p1(tmp_path):
    """Draw a solver's material model: one panel per material parameter.

    An acoustic solver has its velocity alone; an isotropic elastic one its
    density and two wave speeds. The sources, the receivers and the outline
    of the absorbing layer are drawn on every panel.
    """
    wave = get_wave_obj()
    filename = tmp_path / "model_p1.png"
    corners = [(-0.25, 0.25), (-0.75, 0.25), (-0.75, 0.75), (-0.25, 0.75)]
    figure = spyro.plots.plot_model_in_p1(
        wave, dx=0.05, filename=filename, abc_points=corners,
    )
    assert filename.exists()
    panel, colorbar = figure.axes
    assert panel.get_title() == "$c_p$"
    assert colorbar.get_ylabel() == "km/s"
    # The markers of the sources and receivers, then the closed outline.
    outline = panel.lines[-1]
    assert len(panel.lines) == 1 + len(wave.source_locations) + len(wave.receiver_locations)
    assert np.allclose(outline.get_xydata()[[0, -1]], [[0.25, -0.25]] * 2)

    with pytest.warns(DeprecationWarning, match="flip_axis"):
        spyro.plots.plot_model_in_p1(wave, dx=0.05, filename=None, flip_axis=False)

    # An elastic medium, whose model comes from the input dictionary and
    # has not been built by a forward solve yet.
    from tests.on_one_core.test_fwi_automated_adjoint import (
        ELASTIC_GUESS, build_elastic_dictionary,
    )
    elastic = spyro.IsotropicWave(dictionary=build_elastic_dictionary(ELASTIC_GUESS))
    elastic.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    figure = spyro.plots.plot_model_in_p1(elastic, dx=0.05, filename=None)
    # The panels come first in the figure, then their colour bars.
    panels, colorbars = figure.axes[:3], figure.axes[3:]
    assert [panel.get_title() for panel in panels] == [r"$\rho$", "$c_p$", "$c_s$"]
    assert [colorbar.get_ylabel() for colorbar in colorbars] == [
        "g/cm$^3$", "km/s", "km/s",
    ]


def test_plot_model_high_resolution(tmp_path):
    """Draw the velocity model regridded on a fine P1 mesh."""
    wave = get_wave_obj()
    filename = tmp_path / "model_fine.png"
    spyro.plots.plot_model(
        wave, filename=str(filename), high_resolution=True,
        high_resolution_grid_value=0.05,
    )
    assert filename.exists()


@pytest.mark.newer_firedrake
def test_plot_scalar_field(tmp_path):
    """Draw two fields side by side, sampled on a grid, depth downwards.

    The fields are sampled with Firedrake's ``PointEvaluator``, which older
    Firedrake versions do not have.
    """
    import firedrake as fire

    mesh = fire.RectangleMesh(4, 4, 1.0, 2.0, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] *= -1.0   # depth is negative, as in spyro
    V = fire.FunctionSpace(mesh, "CG", 2)
    z, x = fire.SpatialCoordinate(mesh)
    velocity = fire.Function(V).interpolate(1.5 - z)
    gradient = fire.Function(V).interpolate(x - 1.0)

    figure = spyro.plots.plot_scalar_field(
        [velocity, gradient],
        tmp_path / "fields.png",
        titles=["velocity", "gradient"],
        vmin=[1.5, -1.0],
        vmax=[2.5, 1.0],
        colorbar_label=["km/s", "s$^2$/km"],
        sources=[(-0.1, 1.0)],
        # As create_transect gives them: an array, with no truth value.
        receivers=create_transect((-0.9, 0.5), (-0.9, 1.5), 5),
        outline=[(-0.2, 0.2), (-0.8, 0.2), (-0.8, 1.8), (-0.2, 1.8)],
        spacing=0.05,
    )
    assert (tmp_path / "fields.png").exists()
    # One panel per field, each with its own colour bar and label; the
    # panels come first in the figure, then the colour bars.
    assert len(figure.axes) == 4
    assert [axis.get_ylabel() for axis in figure.axes[2:]] == ["km/s", "s$^2$/km"]
    # A source, five receivers and the outline on each panel.
    assert all(len(panel.lines) == 7 for panel in figure.axes[:2])

    # Three panels on two columns: two rows, the last slot left empty.
    figure = spyro.plots.plot_scalar_field(
        [velocity, gradient, velocity], tmp_path / "grid.png", columns=2,
        spacing=0.05,
    )
    assert (tmp_path / "grid.png").exists()
    visible = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible) == 6 and len(figure.axes) == 7

    with pytest.raises(ValueError):
        spyro.plots.plot_scalar_field([velocity, gradient], titles=["one"])
    with pytest.raises(ValueError):
        spyro.plots.plot_scalar_field(fire.Function(fire.VectorFunctionSpace(mesh, "CG", 1)))


@pytest.mark.newer_firedrake
@pytest.mark.parametrize("spacing", [0.26, 0.3, 3.0])
def test_scalar_plot_sampling_covers_mesh(spacing: float) -> None:
    """Sample both mesh edges even when spacing does not divide its size.

    Parameters
    ----------
    spacing : float
        Maximum spacing, including a value larger than the domain.

    Returns
    -------
    None
        Assertions check sample locations and plotted values.
    """
    import firedrake as fire
    from spyro.plots.general_plots import _domain_grid

    mesh = fire.RectangleMesh(2, 2, 1.0, 2.0, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    points, layout = _domain_grid(mesh, spacing)
    assert np.allclose(points.min(axis=0), [-1.0, 0.0])
    assert np.allclose(points.max(axis=0), [0.0, 2.0])
    assert layout[:2] == (
        max(2, int(np.ceil(1.0 / spacing)) + 1),
        max(2, int(np.ceil(2.0 / spacing)) + 1),
    )

    space = fire.FunctionSpace(mesh, "CG", 1)
    z, x = fire.SpatialCoordinate(mesh)
    field = fire.Function(space).interpolate(2.0 * z + x)
    figure = spyro.plots.plot_scalar_field(field, spacing=spacing)
    samples = figure.axes[0].images[0].get_array()
    assert not np.any(np.ma.getmaskarray(samples))
    assert np.allclose(samples.ravel(), 2.0 * points[:, 0] + points[:, 1])


@pytest.mark.parametrize("spacing", [0.0, -0.1, np.nan, np.inf])
def test_scalar_plot_rejects_invalid_spacing(spacing: float) -> None:
    """Reject invalid spacing before accessing the mesh.

    Parameters
    ----------
    spacing : float
        Invalid sampling distance.

    Returns
    -------
    None
        An assertion checks the validation error.
    """
    from spyro.plots.general_plots import _domain_grid

    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        _domain_grid(None, spacing)


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
