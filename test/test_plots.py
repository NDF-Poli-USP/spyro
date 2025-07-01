import spyro
from spyro import create_transect
import pytest
import os


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
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    return spyro.examples.Camembert_acoustic(dictionary=dictionary)


def test_plot():
    rectangle_dictionary = {}
    rectangle_dictionary["mesh"] = {
        "Lz": 0.75,  # depth in km - always positive
        "Lx": 1.5,
        "h": 0.05,
    }
    rectangle_dictionary["acquisition"] = {
        "source_locations": [(-0.1, 0.75)],
        "receiver_locations": create_transect((-0.10, 0.1), (-0.10, 1.4), 50),
        "frequency": 8.0,
    }
    rectangle_dictionary["time_axis"] = {
        "final_time": 2.0,  # Final time for event
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(
        dictionary=rectangle_dictionary
    )
    layer_values = [1.5, 3.0]
    z_switches = [-0.5]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    Wave_obj.forward_solve()
    spyro.plots.plot_shots(Wave_obj, show=False, file_name="test_plot", file_format="png")
    expected_file = "test_plot[0].png"
    assert os.path.exists(expected_file)

    spyro.plots.debug_plot(Wave_obj.u_n, filename="test_debug_plot.png")
    expected_file = "test_debug_plot.png"
    assert os.path.exists(expected_file)

    spyro.plots.debug_pvd(Wave_obj.u_n, filename="test_debug_plot1.pvd")
    expected_file = "test_debug_plot1.pvd"
    assert os.path.exists(expected_file)


@pytest.mark.skipif(not is_seismicmesh_installed(), reason="SeismicMesh is not installed")
def test_plot_mesh_sizes():
    mesh_filename = "test_mesh_for_plots.msh"
    Lz = 1.0
    Lx = 2.0
    c = 1.5
    freq = 5.0
    lbda = c/freq
    pad = 0.3
    cpw = 3

    mesh_parameters = {
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
    }

    Mesh_obj = spyro.meshing.AutomaticMesh(
        mesh_parameters=mesh_parameters,
    )
    Mesh_obj.set_seismicmesh_parameters(output_file_name=mesh_filename)
    mesh = Mesh_obj.create_mesh()  # noqa: F841

    output_filename = "mesh_size.png"
    spyro.plots.plot_mesh_sizes(mesh_filename=mesh_filename, output_filename=str(output_filename), show=False)
    assert os.path.exists(str(output_filename))


def test_plot_model_in_p1():
    wave_obj = get_wave_obj()
    filename = "model_p1.png"
    spyro.plots.plot_model_in_p1(wave_obj, filename=str(filename), show=False)
    assert os.path.exists(str(filename))


if __name__ == "__main__":
    test_plot()
    test_plot_mesh_sizes()
    test_plot_model_in_p1()
