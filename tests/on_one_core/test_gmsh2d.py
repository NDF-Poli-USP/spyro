import math
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_functions import AutomaticMesh
import importlib.util

HAS_NUMBA = importlib.util.find_spec("numba") is not None


def test_gmsh2d_structured():
    print("STARTING STRUCTURED MESH TESTS")

    winslow_implementations = ["default", "fast"]
    if HAS_NUMBA:
        print("Numba is installed, making numba test")
        winslow_implementations.append("numba")
    else:
        print("\n[INFO] Numba is not installed. Skipping 'numba' Winslow implementations.")

    structured_configurations = [
        (True, None, 11610),
        (True, "rectangular", 19275),
        (True, "hyperelliptical", 18070),

        (False, None, 11395),
        (False, "rectangular", 19018),
        (False, "hyperelliptical", 17560)
    ]

    for winslow_impl in winslow_implementations:
        for water_interface, padding_type, expected_cells in structured_configurations:

            print(f"\nTesting Struct | Winslow: {winslow_impl:<7} | Water: {str(water_interface):<5} | Padding: {padding_type}")

            avenir_params = {
                "mesh_type": "gmsh_mesh",
                "dimension": 2,

                "length_z": 7760.0,
                "length_x": 32040.0,

                "output_filename": f"avenir_struct_{winslow_impl}_wat_{water_interface}_pad_{padding_type}.msh",

                "velocity_model": "tests/inputfiles/velocity_models/avenir.segy",
                "cells_per_wavelength": 2.0,
                "source_frequency": 3.0,

                "padding_type": padding_type,
                "padding_x": 3000.0,
                "padding_z": 3000.0,
                "hyper_n": 3.0,
                "hmin_segy": 0.0,
                "grade": 0.75,

                "water_interface": water_interface,
                "water_search_value": 0.0,
                "vp_water": 500.0,

                "structured_mesh": True,
                "min_element_size": 150.0,
                "apply_winslow": True,
                "winslow_implementation": winslow_impl,
                "winslow_iterations": 100,
                "winslow_omega": 0.5,
                "extend_segy": False,
                "h_padding": 500.0
            }

            mesh_params = MeshingParameters(input_mesh_dictionary=avenir_params, velocity_model=avenir_params["velocity_model"])
            mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)

            firedrake_mesh = mesh_generator.create_mesh()

            if not hasattr(firedrake_mesh.topology, "_entity_classes") and hasattr(firedrake_mesh, "init"):
                firedrake_mesh.init()

            actual_cells = firedrake_mesh.cell_set.core_size
            print(f"     Cells actual: {actual_cells} | Expected: {expected_cells}")

            assert math.isclose(actual_cells, expected_cells), \
                f"FAILED: Struct | Winslow: {winslow_impl} | Wat: {water_interface} | Pad: {padding_type}. Got {actual_cells}, expected {expected_cells}"


def test_gmsh2d_unstructured():
    print("STARTING UNSTRUCTURED MESH TESTS")

    unstructured_configurations = [
        (True, True, None, 30275),
        (True, True, "rectangular", 37517),
        (True, True, "hyperelliptical", 36380),
        (True, False, None, 30281),
        (True, False, "rectangular", 37549),
        (True, False, "hyperelliptical", 36242),

        (False, True, None, 30275),
        (False, True, "rectangular", 32983),
        (False, True, "hyperelliptical", 32624),
        (False, False, None, 30281),
        (False, False, "rectangular", 32932),
        (False, False, "hyperelliptical", 32502),
    ]

    for extend_segy, water_interface, padding_type, expected_cells in unstructured_configurations:

        print(f"\nTesting Unstruct | Ext SEGY: {str(extend_segy):<5} | Water: {str(water_interface):<5} | Padding: {padding_type}")

        avenir_params = {
            "mesh_type": "gmsh_mesh",
            "dimension": 2,

            "length_z": 7760.0,
            "length_x": 32040.0,
            "output_filename": f"avenir_unstruct_ext_{extend_segy}_wat_{water_interface}_pad_{padding_type}.msh",

            "velocity_model": "tests/inputfiles/velocity_models/avenir.segy",
            "cells_per_wavelength": 2.0,
            "source_frequency": 3.0,

            "padding_type": padding_type,
            "padding_x": 3000.0,
            "padding_z": 3000.0,
            "hyper_n": 3.0,
            "hmin_segy": 0.0,
            "grade": 0.75,

            "water_interface": water_interface,
            "water_search_value": 0.0,
            "vp_water": 500.0,

            "structured_mesh": False,
            "min_element_size": 150.0,
            "apply_winslow": False,
            "winslow_implementation": "default",
            "winslow_iterations": 100,
            "winslow_omega": 0.5,

            "extend_segy": extend_segy,
            "h_padding": 500.0
        }

        mesh_params = MeshingParameters(input_mesh_dictionary=avenir_params, velocity_model=avenir_params["velocity_model"])
        mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)

        firedrake_mesh = mesh_generator.create_mesh()

        if not hasattr(firedrake_mesh.topology, "_entity_classes") and hasattr(firedrake_mesh, "init"):
            firedrake_mesh.init()

        actual_cells = firedrake_mesh.cell_set.core_size
        print(f"     Cells actual: {actual_cells} | Expected: {expected_cells}")

        assert math.isclose(actual_cells, expected_cells), \
            f"FAILED: Unstruct | Ext: {extend_segy} | Wat: {water_interface} | Pad: {padding_type}. Got {actual_cells}, expected {expected_cells}"


def test_gmsh2d_structured_no_winslow():
    print("STARTING STRUCTURED MESH (NO WINSLOW) TESTS")

    configurations = [
        (True, None, 11610),
        (True, "rectangular", 19275),
        (True, "hyperelliptical", 18070),

        (False, None, 11395),
        (False, "rectangular", 19018),
        (False, "hyperelliptical", 17560)
    ]

    for water_interface, padding_type, expected_cells in configurations:

        print(f"\nTesting Struct (No Winslow) | Water: {str(water_interface):<5} | Padding: {padding_type}")

        avenir_params = {
            "mesh_type": "gmsh_mesh",
            "dimension": 2,

            "length_z": 7760.0,
            "length_x": 32040.0,
            "output_filename": f"avenir_struct_nowinslow_wat_{water_interface}_pad_{padding_type}.msh",

            "velocity_model": "tests/inputfiles/velocity_models/avenir.segy",
            "cells_per_wavelength": 2.0,
            "source_frequency": 3.0,

            "padding_type": padding_type,
            "padding_x": 3000.0,
            "padding_z": 3000.0,
            "hyper_n": 3.0,
            "hmin_segy": 0.0,
            "grade": 0.75,

            # Injected water parameter
            "water_interface": water_interface,
            "water_search_value": 0.0,
            "vp_water": 500.0,

            "structured_mesh": True,
            "min_element_size": 150.0,
            "apply_winslow": False,
            "winslow_implementation": "default",
            "winslow_iterations": 100,
            "winslow_omega": 0.5,

            "extend_segy": False,
            "h_padding": 500.0
        }

        mesh_params = MeshingParameters(input_mesh_dictionary=avenir_params, velocity_model=avenir_params["velocity_model"])
        mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)

        firedrake_mesh = mesh_generator.create_mesh()

        if not hasattr(firedrake_mesh.topology, "_entity_classes") and hasattr(firedrake_mesh, "init"):
            firedrake_mesh.init()

        actual_cells = firedrake_mesh.cell_set.core_size
        print(f"     Cells actual: {actual_cells} | Expected: {expected_cells}")

        assert math.isclose(actual_cells, expected_cells), \
            f"FAILED: Struct (No Winslow) | Wat: {water_interface} | Pad: {padding_type}. Got {actual_cells}, expected {expected_cells}"
