import spyro
import firedrake as fire
import math


def test_eikonal_values_fig8():
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "SeismicMesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": spyro.create_transect(
            (-0.10, 0.1), (-0.10, 0.9), 20
        ),
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.00,  # Final time for event
        "dt": 0.0001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/fd_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)

    # Using SeismicMesh:
    cpw = 5.0
    lba = 1.5 / 5.0
    edge_length = lba / cpw
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)

    # Using Firedrake:
    # Lx = 1
    # Lz = 1
    # user_mesh = fire.RectangleMesh(120, 120, Lz, Lx, diagonal="crossed")
    # user_mesh.coordinates.dat.data[:, 0] *= -1.0
    # z, x = fire.SpatialCoordinate(user_mesh)

    # Wave_obj.set_mesh(user_mesh=user_mesh)
    # cond = fire.conditional(x < 0.5, 3.0, 1.5)

    # Rest of setup
    Wave_obj.set_initial_velocity_model(conditional=cond)
    Wave_obj._get_initial_velocity_model()

    Wave_obj.c = Wave_obj.initial_velocity_model
    Wave_obj.forward_solve()

    min_value = Wave_obj.noneikonal_minimum
    max_value = Wave_obj.noneikonal_maximum

    paper_min = 0.085
    paper_max = 0.56

    test_min = math.isclose(min_value, paper_min, rel_tol=0.1)
    test_max = math.isclose(max_value, paper_max, rel_tol=0.2)
    print("min_value: ", min_value)
    print("paper_min: ", paper_min)
    print("max_value: ", max_value)
    print("paper_max: ", paper_max)

    assert all([test_min, test_max])


# Verificar valores das distancias como lref e velocidades
if __name__ == "__main__":
    test_eikonal_values_fig8()
