import spyro
import math
from generate_velocity_model_from_paper import get_paper_velocity


def test_eikonal_values_fig18():
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
        "Lz": 2.4,  # depth in km - always positive
        "Lx": 4.8,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "user_mesh": None,
        "mesh_type": "SeismicMesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.6, 4.8-1.68)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": spyro.create_transect(
            (-0.10, 0.1), (-0.10, 4.0), 20
        ),
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.00,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/figeigteen_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "debug_output": True,
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)

    cpw = 5.0
    lba = 1.5 / 5.0
    edge_length = lba / cpw
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
    V = Wave_obj.function_space
    mesh = Wave_obj.mesh
    c = get_paper_velocity(mesh, V)

    Wave_obj.set_initial_velocity_model(velocity_model_function=c)
    Wave_obj._get_initial_velocity_model()

    Wave_obj.c = Wave_obj.initial_velocity_model
    Wave_obj.forward_solve()

    min_value = Wave_obj.noneikonal_minimum
    paper_min = 0.58755

    # Testing minimum values
    test_min = math.isclose(min_value, paper_min, rel_tol=0.1)
    print("min_value: ", min_value)
    print("paper_min: ", paper_min)
    print(f"Passed the minimum value test: {test_min}")

    # Testing minimum location
    z_min, x_min = Wave_obj.noneikonal_minimum_point
    paper_z_min = -(2.4-1.925)
    paper_x_min = 4.8

    test_z_min = math.isclose(z_min, paper_z_min, rel_tol=0.1)
    test_x_min = math.isclose(x_min, paper_x_min, rel_tol=0.1)

    test_min_point = all([test_z_min, test_x_min])
    print(f"Passed the minimum point location test: {test_min_point}")

    assert all([test_min, test_min_point])


# Verificar valores das distancias como lref e velocidades
if __name__ == "__main__":
    test_eikonal_values_fig18()

# xloc[m] #yloc[m] #c[km/s] #eik[ms]

# hmin = 25m
# 0.000000000000000000e+00	1.925000000000000000e+03	3.700000000000000178e+00	5.841707164493551545e+02
# 1.550000000000000000e+03	0.000000000000000000e+00	2.899999999999999911e+00	7.776865120601910348e+02

# hmin = 18.75m (paper for FR=16)
# 0.000000000000000000e+00	1.893750000000000000e+03	3.700000000000000178e+00	5.875518884462277356e+02
# 1.556250000000000000e+03	0.000000000000000000e+00	2.899999999999999911e+00	7.726379453924002974e+02

# xloc[m] #yloc[m] #c[km/s] #eik[ms] 
# Marmousi hmin = 12.5m
# 9.200000000000000000e+03	1.687500000000000000e+03	3.500000000000000444e+00	1.122692634216587066e+03
# 5.662500000000000000e+03	0.000000000000000000e+00	3.859560000000000102e+00	1.039170607108715558e+03