import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
from spyro.utils.cost import comp_cost


def wave_dict(dt_usu, layer_shape, degree_layer,
              habc_reference_freq, get_ref_model):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary
    get_ref_model : `bool`
        If True, the infinite model is created. If False, the absorbing layer
        is created based on the model parameters.

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    '''

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
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
    }

    # Simulate for 2.0 seconds.
    fr_files = max(int(100 * 0.00040 / dt_usu), 1)
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Integer >= 2. Only for hypershape
        "habc_reference_freq": habc_reference_freq,  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": get_ref_model,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output/forward/fw_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output/preamble/acoustic_potential_energy",
    }

    return dictionary


def preamble_habc(dictionary, edge_length):
    '''
    Run the infinite model and the Rikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km

    Returns
    -------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/MSH_")

    # # ============ EIKONAL ANALYSIS ============
    # # Reference to resource usage
    # tRef = comp_cost("tini")

    # # Initializing Eikonal object
    # Eik_obj = eik.Eikonal(Wave_obj)

    # # Finding critical points
    # Wave_obj.critical_boundary_points(Eik_obj)

    # # Estimating computational resource usage
    # comp_cost("tfin", tRef=tRef,
    #           user_name=Wave_obj.path_save + "preamble/EIK_")

    return Wave_obj


def test_loop_habc():
    '''
    Loop for applying the HABC to the model in Fig. 8 of Salas et al. (2022).
    '''

    case = 0  # Integer from 0 to 3

    # ============ SIMULATION PARAMETERS ============

    # Mesh size
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.05]  # [0.050, 0.040, 0.032, 0.025, 0.020]

    # Timestep size
    dt_usu_lst = [0.00100]  # [0.00100, 0.00080, 0.00064, 0.00040]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = edge_length / 50.
    print("\nMesh Size: {:.3f} km".format(edge_length))
    print("Timestep Size: {:.3f} ms\n".format(1e3 * dt_usu))

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [None]  # [None, 2, 3, 4, 5]

    # Reference frequency
    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    # Infinite model
    get_ref_model = True

    # Loop for HABC cases
    loop_modeling = not get_ref_model

    # Error criterion for heuristic factor xCR
    crit_opt = "error_difference"  # "error_integral"

    # Number of points for regression (odd number)
    n_pts = 1

    # ============ MESH AND EIKONAL ============
    # Create dictionary with parameters for the model
    dictionary = wave_dict(dt_usu, "rectangular",
                           None, "source", get_ref_model)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_habc(dictionary, edge_length)


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_loop_habc()
