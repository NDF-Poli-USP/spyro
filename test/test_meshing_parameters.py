import pytest
import spyro


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
        input_mesh_dictionary = dictionary["mesh"],
        dimension = dictionary["options"]["dimension"],
        source_frequency = dictionary["acquisition"]["frequency"],
        method = "mass_lumped_triangle",
        degree = dictionary["options"]["degree"],
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
    with pytest.raises(ValueError, match="mesh_type must be one of"):
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
    with pytest.raises(ValueError, match="method must be one of"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="invalid_method",
            degree=4,
        )

if __name__ == "__main__":
    test_initialize_mesh_pam()
    test_negative_length_z_raises()
    test_negative_length_x_raises()
    test_negative_length_y_raises()
    test_invalid_mesh_type_raises()
    test_invalid_method_raises()
