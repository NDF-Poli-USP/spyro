import firedrake as fire
from spyro.utils.cost import comp_cost
# import spyro.abc.abc_layer as abc
import spyro.pml.pml_nsnc as pml


def wave_dict(dt_usu, fr_files, habc_reference_freq, get_ref_model):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation
    fr_files : `int`
        Frequency of the output files to be saved in the simulation
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary'
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
        "Lz": 1.,  # depth in km - always positive
        "Lx": 1.,  # width in km - always positive
        "Ly": 0.,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)], # ToDo
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "PML",  # Activate HABC
        "exponent": 2,
        "R": 1e-6,
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
        "acoustic_energy_filename": "output/preamble/acoustic_pot_energy",
    }

    return dictionary


def preamble_pml(dictionary, edge_length, f_est):
    '''
    Run the infinite model and the Eikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`, optional
        Factor for the stabilizing term in Eikonal Eq.

    Returns
    -------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    # Wave_obj = abc.ABC_Layer_Wave(dictionary=dictionary)
    Wave_obj = pml.PML_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations(f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    Wave_obj.critical_boundary_points()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/EIK_")

    return Wave_obj


def pml_fig8(Wave_obj, modal_solver):
    '''
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022)

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
   modal_solver : `str`
        Method to use for solving the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_abc_layer_case()

    # Acquiring reference signal
    Wave_obj.get_reference_signal()

    # Determining layer size
    Wave_obj.layer_size_criterion(n_root=1)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_with_layer()

    # Updating velocity model
    Wave_obj.velocity_abc()

    # Setting the damping profile within absorbing layer
    Wave_obj.pml_layer()

    # Applying NRBCs on outer boundary layer
    Wave_obj.nrbc_on_boundary_layer(sommerfeld_bc=True)

    # Solving the forward problem
    Wave_obj.forward_solve()

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Plotting the solution at receivers and the error measures
    Wave_obj.comparison_plots()


def test_loop_pml_2d():
    '''
    Loop for applying the HABC to the model in Fig. 8 of Salas et al. (2022)
    '''

    case = 0  # Integer from 0 to 4

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.1000, 0.0625, 0.0500, 0.0250, 0.0200]

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu_lst = [0.00100, 0.00064, 0.00050, 0.00032, 0.00020]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.06, 0.02, 0.02, 0.02, 0.04]

    # Maximum divisor of the final time
    max_div_tf_lst = [5, 3, 5, 2, 3]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = dt_usu_lst[case]
    f_est = f_est_lst[case]
    max_div_tf = max_div_tf_lst[case]
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Timestep Size: {:.3f} ms".format(1e3 * dt_usu), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    print("Maximum Divisor of Final Time: {}".format(max_div_tf), flush=True)

    # ============ HABC PARAMETERS ============

    # Infinite model (True: Infinite model, False: HABC scheme)
    get_ref_model = False

    # Loop for HABC cases
    loop_modeling = not get_ref_model

    # Reference frequency
    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    # Modal solver for fundamental frequency
    modal_solver = 'KRYLOVSCH_CH'

    # ============ MESH AND EIKONAL ============
    # Create dictionary with parameters for the model
    fr_files = max(int(100 * max(dt_usu_lst) / dt_usu), 1)
    dictionary = wave_dict(dt_usu, fr_files, "source", get_ref_model)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_pml(dictionary, edge_length, f_est)

    # ============ REFERENCE MODEL ============
    if get_ref_model:
        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing reference get_reference_signal
        Wave_obj.infinite_model(check_dt=True, max_divisor_tf=max_div_tf)

        # Set model parameters for the HABC scheme
        Wave_obj.abc_get_ref_model = False

        # Estimating computational resource usage
        comp_cost("tfin", tRef=tRef,
                  user_name=Wave_obj.path_save + "preamble/INF_")

    # ============ HABC SCHEME ============

    # Data to print on screen
    fref_str = "HABC Reference Frequency: {}"
    mods_str = "Modal Solver for Fundamental Frequency: {}"

    # Loop for different layer shapes and degrees
    for habc_ref_freq in habc_reference_freq_lst:

        # Reference frequency for sizing the hybrid absorbing layer
        Wave_obj.abc_reference_freq = habc_ref_freq
        print(fref_str.format(habc_ref_freq.capitalize()), flush=True)

        # Modal solver for fundamental frequency
        print(mods_str.format(modal_solver), flush=True)

        try:

            # Reference to resource usage
            tRef = comp_cost("tini")

            # Run the HABC scheme
            pml_fig8(Wave_obj, modal_solver)

            # Estimating computational resource usage
            comp_cost("tfin", tRef=tRef,
                      user_name=Wave_obj.path_case_abc)

        except fire.ConvergenceError as e:
            print(f"Error Solving: {e}", flush=True)
            break


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_loop_pml_2d()

# eik_min = 83.333 ms
# f_est   100m   62.5m     50m     25m     20m
#  0.01 66.836   --/--   --/--   --/--   --/--
#  0.02 73.308  83.907* 83.944* 83.812* 82.193
#  0.03 77.178  85.322  85.068  84.398   --/--
#  0.04 79.680  86.352  85.933  84.901  83.434*
#  0.05 81.498  87.263  86.718  85.375  83.863
#  0.06 82.942* 88.130  87.470  85.837  84.250
#  0.07 84.160  88.977  88.207  86.292  84.613
#  0.08 85.233  89.815  88.934  86.745  84.961

# n_hyp  100m  62.5m  50m  25m  20m
# n_min   2.0    2.0  2.0  2.0  2.0
# n_max   4.4    4.8  4.7  4.7  4.6

# freq    N2.0    N3.0    N4.0    N4.4     REC
# num  0.50443 0.48266 0.47423 0.47270 0.45539
# anr  0.53970 0.52214 0.51563 0.51410 0.50036
# anh  0.52300 0.48952 0.49112 0.48952 0.47102

# from time import perf_counter  # For runtime
# tRef = perf_counter()
# print(f"Time: {perf_counter() - tRef:.4f} seconds")
