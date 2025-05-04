import spyro
import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
from os import getcwd
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}


def test_habc_fig8():
    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 4,  # p order p<=4 for 2D
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
        "source_locations": [(-0.5, 0.25)],
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
        "frequency": 5.0,  # in Hz
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

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "hybrid",
        "layer_shape": "rectangular",
        # "layer_shape": "hypershape",
        "degree_layer": 5,
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/fd_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    # Create the acoustic wave object with HABCs
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length = 0.05
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

    if Wave_obj.fwi_iter == 0:
        # Initial velocity model
        cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
        Wave_obj.set_initial_velocity_model(conditional=cond)

        # Preamble mesh operations
        Wave_obj.preamble_mesh_operations(p_usu=2)

        # Initializing Eikonal object
        Eik_obj = eik.Eikonal(Wave_obj)
        histPcrit = None

    # Determining layer size
    Wave_obj.size_habc_criterion(Eik_obj, histPcrit,
                                 layer_based_on_mesh=True)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Setting the damping profile within absorbing layer
    Wave_obj.damping_layer()

    # Applying NRBCs on outer boundary layer
    Wave_obj.cos_ang_HigdonBC()

    # Solving the forward problem
    Wave_obj.forward_solve()
    ipdb.set_trace()


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_habc_fig8()
