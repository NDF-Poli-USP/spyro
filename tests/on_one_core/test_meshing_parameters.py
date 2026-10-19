import pytest
import spyro
from spyro.meshing.meshing_parameters import cells_per_wavelength
import warnings
import numpy as np
from pathlib import Path
import os


def test_initialize_mesh_pam():
    """Test that `MeshingParameters` stores the values given at construction."""
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
        "length_z": 3.0,  # depth in km - always positive
        "length_x": 3.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
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
    test_unit = (meshing_pam.unit == "km")
    test_length_z = (meshing_pam.length_z == dictionary["mesh"]["length_z"])
    test_length_x = (meshing_pam.length_z == dictionary["mesh"]["length_x"])
    test_length_y = (meshing_pam.length_y == dictionary["mesh"]["length_y"])
    test_degree = (meshing_pam.degree == dictionary["options"]["degree"])
    test_source_frequency = (meshing_pam.source_frequency == dictionary[
        "acquisition"]["frequency"])
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
    """Test that a negative `length_z` raises `ValueError`."""
    dictionary = {
        "mesh": {
            "length_z": -1.0,
            "length_x": 3.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="'length_z' must be greater than or equal to 0"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_negative_length_x_raises():
    """Test that a negative `length_x` raises `ValueError`."""
    dictionary = {
        "mesh": {
            "length_z": 3.0,
            "length_x": -2.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="'length_x' must be greater than or equal to 0"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_negative_length_y_raises():
    """Test that a negative `length_y` raises `ValueError`."""
    dictionary = {
        "mesh": {
            "length_z": 3.0,
            "length_x": 3.0,
            "length_y": -0.5,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        }
    }
    with pytest.raises(ValueError, match="'length_y' must be greater than or equal to 0"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_invalid_mesh_type_raises():
    """Test that an unsupported `mesh_type` raises `ValueError`."""
    dictionary = {
        "mesh": {
            "length_z": 3.0,
            "length_x": 3.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "invalid_mesh_type",
        }
    }
    with pytest.raises(ValueError, match="Invalid mesh_type"):
        spyro.meshing.MeshingParameters(
            input_mesh_dictionary=dictionary["mesh"],
            dimension=2,
            source_frequency=5.0,
            method="mass_lumped_triangle",
            degree=4,
        )


def test_invalid_method_raises():
    """Test that an unsupported `method` raises `ValueError`."""
    dictionary = {
        "mesh": {
            "length_z": 3.0,
            "length_x": 3.0,
            "length_y": 0.0,
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
    """Test that `cells_per_wavelength` returns tabulated values or `None`."""
    assert cells_per_wavelength('mass_lumped_triangle', 2, 2) == 7.02
    assert cells_per_wavelength('mass_lumped_triangle', 3, 2) == 3.70
    assert cells_per_wavelength('spectral_quadrilateral', 2, 2) is None


def test_cells_per_wavelength_unknown_key():
    """Test that `cells_per_wavelength` returns `None` for an unknown method."""
    assert cells_per_wavelength('unknown', 1, 1) is None


def test_meshing_parameters_init_defaults():
    """Test the default attribute values of a `MeshingParameters` without input."""
    mp = spyro.meshing.MeshingParameters()
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
    """Test that `mesh_file` rejects invalid or missing files and accepts a .msh."""
    mp = spyro.meshing.MeshingParameters()
    # Non-string or wrong extension
    with pytest.raises(TypeError):
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
    """Test that mixing meter and km lengths warns that a length is in km."""
    mp = spyro.meshing.MeshingParameters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.length_z = 200  # meters
        mp.length_x = 0.1  # km, triggers warning
        assert any("appears to be in km" in str(warn.message) for warn in w)


def test_source_frequency_warnings():
    """Test that very low and very high source frequencies emit warnings."""
    mp = spyro.meshing.MeshingParameters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.source_frequency = 1.0
        mp.source_frequency = 100.0
        assert any("too low" in str(warn.message) for warn in w)
        assert any("too high" in str(warn.message) for warn in w)


def test_cells_per_wavelength_and_edge_length_mutual_exclusion():
    """Test that setting `edge_length` warns and clears `cells_per_wavelength`."""
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
    """Test that `periodic` can be enabled only for `firedrake_mesh`."""
    mp = spyro.meshing.MeshingParameters()
    mp.mesh_type = "firedrake_mesh"
    mp.periodic = True
    assert mp.periodic is True
    with pytest.raises(ValueError, match="Periodic meshes"):
        mp.mesh_type = "user_mesh"
    mp.periodic = False
    mp.mesh_type = "user_mesh"
    with pytest.raises(ValueError, match="Periodic meshes"):
        mp.periodic = True


def _gmsh_parameters(**mesh_options) -> spyro.meshing.MeshingParameters:
    """Build 2D `gmsh_mesh` parameters with extra mesh options.

    Parameters
    ----------
    **mesh_options
        Entries added to, or overriding, the base mesh dictionary.

    Returns
    -------
    spyro.meshing.MeshingParameters
        The constructed meshing parameters.
    """
    mesh_dictionary = {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 0.0,
        "mesh_file": None,
        "mesh_type": "gmsh_mesh",
        "edge_length": 0.1,
        "h_padding": 0.1,
        **mesh_options,
    }
    return spyro.meshing.MeshingParameters(
        input_mesh_dictionary=mesh_dictionary,
        dimension=2,
        source_frequency=5.0,
    )


def test_structured_and_unstructured_quad_are_mutually_exclusive():
    """Test that enabling both quad mesh options raises `ValueError`."""
    with pytest.raises(
        ValueError,
        match="'structured_mesh' and 'unstructured_quad_mesh' are mutually exclusive",
    ):
        _gmsh_parameters(structured_mesh=True, unstructured_quad_mesh=True)


@pytest.mark.parametrize(
    "structured_mesh, unstructured_quad_mesh",
    [(False, False), (True, False), (False, True)],
)
def test_single_quad_mesh_option_is_accepted(structured_mesh, unstructured_quad_mesh):
    """Test that at most one enabled quad mesh option is accepted.

    Parameters
    ----------
    structured_mesh : bool
        Value of the `structured_mesh` option.
    unstructured_quad_mesh : bool
        Value of the `unstructured_quad_mesh` option.
    """
    mp = _gmsh_parameters(
        structured_mesh=structured_mesh,
        unstructured_quad_mesh=unstructured_quad_mesh,
    )
    assert mp.gmsh.structured_mesh is structured_mesh
    assert mp.gmsh.unstructured_quad_mesh is unstructured_quad_mesh


def _mp(**mesh_dictionary):
    """Build MeshingParameters from keyword arguments used as the input dictionary.

    Parameters
    ----------
    **mesh_dictionary
        Entries of the input mesh dictionary.

    Returns
    -------
    spyro.meshing.MeshingParameters
        The constructed object.
    """
    return spyro.meshing.MeshingParameters(input_mesh_dictionary=mesh_dictionary)


def test_velocity_model_keyword_is_stored():
    velocity_model = {"vp_values": [[1.5]], "grid_spacing": 0.1}
    mp = spyro.meshing.MeshingParameters(velocity_model=velocity_model)
    assert mp.velocity_model is velocity_model


def test_segy_deprecation_only_warns_when_set():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _mp(mesh_type="gmsh_mesh")
    with pytest.warns(UserWarning, match="deprecated"):
        mp = _mp(mesh_type="gmsh_mesh", segy_velocity_model="vel.segy")
    assert mp.velocity_model == "vel.segy"


def test_automatic_mesh_follows_mesh_type():
    mp = spyro.meshing.MeshingParameters()
    assert mp.automatic_mesh is False
    mp.mesh_type = "gmsh_mesh"
    assert mp.automatic_mesh is True
    mp.mesh_type = "user_mesh"
    assert mp.automatic_mesh is False


def test_all_input_keys_are_attributes():
    mp = spyro.meshing.MeshingParameters()
    for key in mp.INPUT_KEYS:
        assert hasattr(mp, key) or hasattr(mp.gmsh, key), key


def test_gmsh_options_from_dictionary():
    mp = _mp(mesh_type="gmsh_mesh", grade=0.5, winslow_omega=None)
    assert mp.gmsh.grade == 0.5
    assert mp.gmsh.winslow_omega == 0.5
    assert not hasattr(mp, "grade")


@pytest.mark.parametrize("dictionary", [{"quadrilateral": 1}, {"apply_winslow": 0}, {"periodic": 1}])
def test_boolean_options_reject_integers(dictionary):
    with pytest.raises(TypeError, match="must be a boolean"):
        spyro.meshing.MeshingParameters(input_mesh_dictionary=dictionary)


def test_validate_catches_direct_assignment():
    mp = spyro.meshing.MeshingParameters()
    mp.gmsh.winslow_omega = 2.0
    with pytest.raises(ValueError, match="winslow_omega"):
        mp.validate()


def test_gmsh_options_exist_for_every_mesh_type():
    mp = _mp(mesh_type="firedrake_mesh")
    assert mp.gmsh.vp_water is None
    assert mp.gmsh.winslow_iterations == 5000
    mp.set_mesh(input_mesh_parameters={"mesh_type": "gmsh_mesh", "grade": 0.5})
    assert mp.gmsh.grade == 0.5


def test_set_mesh_rejects_unknown_keys():
    mp = spyro.meshing.MeshingParameters()
    with pytest.raises(ValueError, match="did you mean 'edge_length'"):
        mp.set_mesh(input_mesh_parameters={"edge_lenght": 0.2})


def test_set_mesh_is_order_independent():
    mp = spyro.meshing.MeshingParameters()
    mp.set_mesh(input_mesh_parameters={"periodic": True, "mesh_type": "firedrake_mesh"})
    assert mp.periodic is True


def test_set_mesh_none_leaves_value_unchanged():
    mp = _mp(length_z=1.0)
    mp.set_mesh(input_mesh_parameters={"length_z": None})
    assert mp.length_z == 1.0


def test_set_mesh_keyword_pad_wins():
    mp = spyro.meshing.MeshingParameters()
    with pytest.warns(UserWarning, match="abc_pad_length"):
        mp.set_mesh(input_mesh_parameters={"abc_pad_length": 0.2}, abc_pad_length=0.3)
    assert mp.abc_pad_length == 0.3


def test_zero_pad_does_not_fix_unit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mp = _mp(abc_pad_length=0.0, length_z=2000.0, length_x=3000.0, padding_x=50.0)
    assert mp.unit == "meters"


def test_large_pad_in_km_domain_warns():
    with pytest.warns(UserWarning, match="appears to be in meters"):
        _mp(length_z=2.0, length_x=3.0, abc_pad_length=500.0)


def test_explicit_unit():
    with pytest.warns(UserWarning, match="appears to be in km"):
        mp = _mp(unit="m", length_z=2.0)
    assert mp.unit == "meters"
    with pytest.raises(ValueError, match="Invalid unit"):
        _mp(unit="miles")


def test_unit_aware_gmsh_defaults():
    km = _mp(length_z=2.0, length_x=3.0)
    meters = _mp(length_z=2000.0, length_x=3000.0)
    assert km.h_padding == pytest.approx(0.5)
    assert meters.h_padding == pytest.approx(500.0)
    assert km.min_element_size == pytest.approx(0.035)
    assert meters.min_element_size == pytest.approx(35.0)
    assert _mp(length_z=2.0, h_padding=0.1).h_padding == 0.1


def test_source_frequency_types():
    mp = spyro.meshing.MeshingParameters()
    mp.source_frequency = np.int64(5)
    assert mp.source_frequency == 5
    with pytest.raises(TypeError):
        mp.source_frequency = True
    with pytest.raises(ValueError):
        mp.source_frequency = -1.0


def test_edge_length_validation():
    mp = spyro.meshing.MeshingParameters()
    with pytest.raises(ValueError, match="'edge_length' must be greater than 0"):
        mp.edge_length = -1
    mp.cells_per_wavelength = 3.0
    with pytest.warns(UserWarning, match="Mutual exclusion"):
        mp.edge_length_x = 0.1
    assert mp.cells_per_wavelength is None
    with pytest.warns(UserWarning, match="removes edge_length"):
        mp.cells_per_wavelength = 3.0
    assert mp.edge_length_x is None


def test_edge_length_along():
    mp = _mp(edge_length=0.1, edge_length_x=0.2)
    assert mp.edge_length_along("x") == 0.2
    assert mp.edge_length_along("z") == 0.1
    with pytest.raises(ValueError):
        mp.edge_length_along("w")


def test_grid_velocity_data_read_from_dictionary():
    data = {"vp_values": [[1.5]], "grid_spacing": 0.1}
    assert _mp(grid_velocity_data=data).grid_velocity_data is data


def test_keyword_beats_dictionary():
    with pytest.warns(UserWarning, match="'degree' was given both"):
        mp = spyro.meshing.MeshingParameters(degree=4, input_mesh_dictionary={"degree": 2})
    assert mp.degree == 4
    mp = _mp(quadrilateral=True, method="spectral_quadrilateral", negative_z=False)
    assert mp.quadrilateral is True
    assert mp.method == "spectral_quadrilateral"
    assert mp.negative_z is False


def test_output_filename_validation(tmp_path):
    mp = spyro.meshing.MeshingParameters()
    with pytest.raises(ValueError, match="output_filename"):
        mp.output_filename = "mesh.txt"
    mp.output_filename = tmp_path / "mesh.msh"
    assert mp.output_filename == str(tmp_path / "mesh.msh")


def test_completeness():
    mp = _mp(mesh_type="firedrake_mesh", edge_length=0.1)
    assert not mp.is_complete
    assert "'firedrake_mesh' requires 'length_z'." in mp.missing_parameters()
    mp.length_z = 1.0
    mp.length_x = 1.0
    assert mp.is_complete
    assert mp.check_completeness(verbose=False)
    with pytest.warns(UserWarning, match="removes edge_length"):
        mp.set_mesh(input_mesh_parameters={"cells_per_wavelength": 3.0})
    assert mp.missing_parameters() == [
        "'cells_per_wavelength' requires 'source_frequency'.",
        "'cells_per_wavelength' requires 'minimum_velocity', "
        "'grid_velocity_data', or 'velocity_model'.",
    ]


def test_init_does_not_print(capsys):
    _mp(mesh_type="firedrake_mesh")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("dictionary", [
    {"winslow_omega": 2.0},
    {"winslow_iterations": -1},
    {"grade": 0.0},
    {"gmsh_num_threads": 0},
    {"segy_byte_order": "middle"},
    {"segy_axes_order_sort": "A"},
    {"segy_axes_order": (0, 0, 1)},
    {"segy_dtype": "not_a_dtype"},
    {"structured_mesh": True, "unstructured_quad_mesh": True},
    {"unstructured_quad_mesh": True, "dimension": 3},
    {"dimension": 4},
    {"degree": 0},
    {"method": "spectral_quadrilateral"},
    {"method": "mass_lumped_triangle", "quadrilateral": True},
    {"mesh_type": "SeismicMesh", "quadrilateral": True},
])
def test_invalid_parameters_raise(dictionary):
    with pytest.raises(ValueError):
        spyro.meshing.MeshingParameters(input_mesh_dictionary=dictionary)


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
