import pytest
import spyro
from spyro.meshing.meshing_parameters import cells_per_wavelength
import warnings
from pathlib import Path
import os


def test_initialize_mesh_pam():
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    dictionary["mesh"] = {
        "Lz": 3.0,  # depth in km - always positive
        "Lx": 3.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.5, 1.5)],
        "frequency": 5.0,
        "delay": 0.3,
        "receiver_locations": [(-1.5, 2.0)],
        "delay_type": "time",
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.0,  # Final time for event
        "dt": 0.001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }
    meshing_pam = spyro.meshing.MeshingParameters(
        input_mesh_dictionary=dictionary["mesh"],
        dimension=dictionary["options"]["dimension"],
        source_frequency=dictionary["acquisition"]["frequency"],
        method="mass_lumped_triangle",
        degree=dictionary["options"]["degree"],
    )

    # Testing correct values:
    test_unit = (meshing_pam._unit == "km")
    test_length_z = (meshing_pam.length_z == dictionary["mesh"]["Lz"])
    test_length_x = (meshing_pam.length_z == dictionary["mesh"]["Lx"])
    test_length_y = (meshing_pam.length_y == dictionary["mesh"]["Ly"])
    test_degree = (meshing_pam.degree == dictionary["options"]["degree"])
    test_source_frequency = (meshing_pam.source_frequency == dictionary["acquisition"]["frequency"])
    test_mesh_type = (meshing_pam.mesh_type == "firedrake_mesh")
    test_method = (meshing_pam.method == "mass_lumped_triangle")

    assert test_unit, "Expected unit 'km'"
    assert test_length_z, "Expected length_z to match input"
    assert test_length_x, "Expected length_x to match input"
    assert test_degree, "Expected degree to match input"
    assert test_mesh_type, "Expected mesh_type 'firedrake_mesh'"
    assert test_method, "Expected method 'mass_lumped_triangle'"
    assert test_source_frequency, "Expected source_frequency to match input"
    assert test_length_y, "Expected length_y to match input"

    print("END")


def test_negative_length_z_raises():
    dictionary = {
        "mesh": {
            "Lz": -1.0,
            "Lx": 3.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="Please do not use negative value for _length_z"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_negative_length_x_raises():
    dictionary = {
        "mesh": {
            "Lz": 3.0,
            "Lx": -2.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="Please do not use negative value for _length_x"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_negative_length_y_raises():
    dictionary = {
        "mesh": {
            "Lz": 3.0,
            "Lx": 3.0,
            "Ly": -0.5,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="Please do not use negative value for _length_y"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_invalid_mesh_type_raises():
    dictionary = {
        "mesh": {
            "Lz": 3.0,
            "Lx": 3.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "invalid_mesh_type",
        }
    }
    with pytest.raises(ValueError, match="Invalid mesh_type: 'invalid_mesh_type'."):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_invalid_method_raises():
    dictionary = {
        "mesh": {
            "Lz": 3.0,
            "Lx": 3.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="Invalid method: 'invalid_method'."):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="invalid_method",
            degree=4,
        )


def test_cells_per_wavelength_known_key():
    assert cells_per_wavelength('mass_lumped_triangle', 2, 2) == 7.02
    assert cells_per_wavelength('mass_lumped_triangle', 3, 2) == 3.70
    assert cells_per_wavelength('spectral_quadrilateral', 2, 2) is None


def test_cells_per_wavelength_unknown_key():
    assert cells_per_wavelength('unknown', 1, 1) is None


def test_meshing_parameters_init_defaults():
    mp = spyro.meshing.MeshingParameters()
    assert mp.input_mesh_dictionary == {}
    assert mp.dimension is None
    assert mp.mesh_type is None
    assert mp.periodic is False
    assert mp.mesh_file is None
    assert mp.length_z is None
    assert mp.length_x is None
    assert mp.length_y is None
    assert mp.user_mesh is None
    assert mp.source_frequency is None
    assert mp.abc_pad_length is None
    assert mp.quadrilateral is False
    assert mp.method is None
    assert mp.degree is None
    assert mp.minimum_velocity is None
    assert mp.velocity_model is None
    assert mp.automatic_mesh is False


def test_mesh_file_validation():
    mp = spyro.meshing.MeshingParameters()
    # Non-string or wrong extension
    with pytest.raises(ValueError):
        mp.mesh_file = 123
    with pytest.raises(ValueError):
        mp.mesh_file = "mesh.txt"
    # File does not exist
    with pytest.raises(FileNotFoundError):
        mp.mesh_file = "not_exist.msh"
    # File exists
    mesh_path = Path.cwd() / "mesh.msh"
    mesh_path.write_text("dummy")
    mp.mesh_file = str(mesh_path)
    if mesh_path.exists():
        os.remove(mesh_path)
    assert mp.mesh_file == str(mesh_path)


def test_length_unit_check_warns():
    mp = spyro.meshing.MeshingParameters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.length_z = 200  # meters
        mp.length_x = 0.1  # km, triggers warning
        assert any("appears to be in km" in str(warn.message) for warn in w)


def test_source_frequency_warnings():
    mp = spyro.meshing.MeshingParameters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.source_frequency = 1.0
        mp.source_frequency = 100.0
        assert any("too low" in str(warn.message) for warn in w)
        assert any("too high" in str(warn.message) for warn in w)


def test_cells_per_wavelength_and_edge_length_mutual_exclusion():
    mp = spyro.meshing.MeshingParameters()
    mp._edge_length = None
    mp._cells_per_wavelength = None
    mp.cells_per_wavelength = 10
    assert mp.cells_per_wavelength == 10
    assert mp.edge_length is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.edge_length = 5
        assert mp.edge_length == 5
        assert mp.cells_per_wavelength is None
        assert any("Mutual exclusion" in str(warn.message) for warn in w)


def test_periodic_only_for_firedrake_mesh():
    mp = spyro.meshing.MeshingParameters()
    mp.mesh_type = "firedrake_mesh"
    mp.periodic = True
    assert mp.periodic is True
    mp.mesh_type = "user_mesh"
    with pytest.raises(ValueError):
        mp.periodic = True


if __name__ == "__main__":
    test_initialize_mesh_pam()
    test_negative_length_z_raises()
    test_negative_length_x_raises()
    test_negative_length_y_raises()
    test_invalid_mesh_type_raises()
    test_invalid_method_raises()
    test_cells_per_wavelength_known_key()
    test_cells_per_wavelength_unknown_key()
    test_meshing_parameters_init_defaults()
    test_mesh_file_validation()
    test_length_unit_check_warns()
    test_source_frequency_warnings()
    test_cells_per_wavelength_and_edge_length_mutual_exclusion()
    test_periodic_only_for_firedrake_mesh()
