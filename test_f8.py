import spyro
import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
import spyro.plots.plots as plt_spyro
from habc import comp_cost
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
        "type": "automatic",
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
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
    }

    # Simulate for 1.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 3.,    # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "hybrid",
        # "layer_shape": "rectangular",
        "layer_shape": "hypershape",  # Options: rectangular or hypershape
        "degree_layer": 2,  # Integer >= 2. Only for "hypershape"
        "habc_reference_freq": "boundary",
        # "habc_reference_freq": "source" , # Options: source or boundary
        "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/fd_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,
        "acoustic_energy_filename": "results/acoustic_potential_energy",
    }

    # Create the acoustic wave object with HABCs
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length = 0.05
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations(p_usu=2)

    # Initializing Eikonal object
    Eik_obj = eik.Eikonal(Wave_obj)

    # Finding critical points
    Wave_obj.critical_boundary_points(Eik_obj)

    # Computing reference signal
    Wave_obj.infinite_model()

    # Determining layer size
    Wave_obj.size_habc_criterion(crtCR=1, layer_based_on_mesh=True)

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

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Plotting the solution at receivers
    plt_spyro.plot_hist_receivers(Wave_obj)


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":

    # Reference to resource usage
    tRef = comp_cost('tini')

    # Run the test for Fig. 8
    test_habc_fig8()

    # Estimating computational resource usage
    comp_cost('tfin', tRef=tRef)
