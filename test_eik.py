import spyro
import firedrake as fire
import ipdb
import spyro.habc.eik as eik
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}


def test_eikonal_values_fig8():
    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 1,  # p order p=4 ok
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "spatial",
    }

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at a microphone
    # near the top of the domain. This transect of receivers is created with
    # the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
        "source_locations": [(-0.5, 0.25)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": spyro.create_transect(
            (-0.10, 0.1), (-0.10, 0.9), 20),
    }

    # Simulate for 1.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.00,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/fd_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    # Create the acoustic wave object
    Wave_obj = spyro.AcousticWave(dictionary=dictionary)

    # Mesh
    edge_length = 0.05
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)

    # Rest of setup
    p = dictionary["options"]["degree"]
    Wave_obj.function_space = fire.FunctionSpace(Wave_obj.mesh, 'CG', p)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    Wave_obj.c = Wave_obj.initial_velocity_model
    outfile = fire.VTKFile("/mnt/d/spyro/output/output.pvd")
    outfile.write(Wave_obj.c)

    # Solving Eikonal
    Eik_obj = eik.eikonal(Wave_obj)
    Eik_obj.solve_eik(Wave_obj)

    # Identifying critical points
    Eik_obj.ident_crit_eik(Wave_obj)


# Cheking Eikonal values
if __name__ == "__main__":
    test_eikonal_values_fig8()
