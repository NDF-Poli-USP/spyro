import firedrake as fire
from spyro.utils.cost import comp_cost
import spyro.pml.pml_nsnc as pml


def wave_dict_2d(dt_usu, fr_files, habc_reference_freq, get_ref_model):
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
        "acoustic_energy_filename": "output/pml_test2d/preamble/acoustic_pot_energy",
    }

    return dictionary


def wave_dict_3d(dt_usu, fr_files, habc_reference_freq,
                 get_ref_model, degree_eikonal):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation
    fr_files : `int`
        Frequency of the output files to be saved in the simulation
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary
    get_ref_model : `bool`
        If True, the infinite model is created. If False, the absorbing layer
        is created based on the model parameters.
    degree_eikonal : `int`
        Finite element order for the Eikonal equation. Should be 1 or 2.

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
        "degree": 3,  # p order p<=3 for 3D
        "dimension": 3,  # dimension
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
    Lz, Lx, Ly = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25, 0.5)],
        "frequency": 5.0,  # in Hz
        "delay": 1. / 3.,
        "delay_type": "time",  # "multiples_of_minimum" or "time"
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                               (0., 0., 0), (0., Lx, 0.),
                               (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 1.,    # Final time for event
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
        "degree_eikonal": degree_eikonal,  # Finite element order for the Eikonal analysis
        "get_ref_model": get_ref_model,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output/forward/fw_output_3d.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output/pml_test3d/preamble/acoustic_pot_energy_3d",
    }

    return dictionary


def preamble_pml(dictionary, edge_length, f_est, dimension):
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
    dimension : `int`
        Dimension of the model (2 or 3)

    Returns
    -------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    Wave_obj = pml.PML_Wave(dictionary=dictionary,
                            bc_boundary_pml="Dirichlet",
                            output_folder=f"output/pml_test{dimension}d")

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


def run_reference(Wave_obj, max_divisor_tf=1):
    '''
    Run the infinite model to get the reference signal for the PML scheme

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    max_divisor_tf : `int`, optional
        Index to select the maximum divisor of the final time, converted
        to an integer according to the order of magnitude of the timestep
        size. The timestep size is set to the divisor, given by the index
        in descending order, less than or equal to the user's timestep
        size. If the value is 1, the timestep size is set as the maximum
        divisor. Default is 1

    Returns
    -------
    None
    '''

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Computing reference get_reference_signal
    Wave_obj.infinite_model(check_dt=True, max_divisor_tf=max_divisor_tf)

    # Set model parameters for the HABC scheme
    Wave_obj.abc_get_ref_model = False

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/INF_")


def pml_fig8(Wave_obj, modal_solver):
    '''
    Apply the PML scheme to the model in Fig. 8 of Salas et al. (2022)

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
    Wave_obj.identify_abc_layer_case(
        output_folder=f"output/pml_test{Wave_obj.dimension}d")

    # Acquiring reference signal
    Wave_obj.get_reference_signal()

    # Determining layer size
    Wave_obj.layer_size_criterion(n_root=1)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_with_layer()

    # Updating velocity model
    Wave_obj.velocity_abc()

    # Building the PML layer (damping and BCs)
    Wave_obj.pml_layer()

    # Solving the forward problem
    Wave_obj.forward_solve()

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Plotting the solution at receivers and the error measures
    Wave_obj.comparison_plots()


def run_pml(Wave_obj, habc_reference_freq_lst, modal_solver):
    '''
    Run the PML scheme for different reference frequencies.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    habc_reference_freq_lst : `list`
        List of reference frequencies for sizing the PML layer.
        Options: 'source' or 'boundary'
    modal_solver : `str`
        Method to use for solving the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'

    Returns
    -------
    None
    '''

    # Data to print on screen
    fref_str = "PML Reference Frequency: {}"
    mods_str = "Modal Solver for Fundamental Frequency: {}"

    # Loop for different reference frequencies
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


def test_loop_pml_2d():
    '''
    Loop for applying the PML2D to the model in Fig. 8 of Salas et al. (2022)
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
    max_div_tf_lst = [5, 3, 5, 2, 3]  # Approximate eigenvalue

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = dt_usu_lst[case]
    f_est = f_est_lst[case]
    max_div_tf = max_div_tf_lst[case]
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Timestep Size: {:.3f} ms".format(1e3 * dt_usu), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    print("Maximum Divisor of Final Time: {}".format(max_div_tf), flush=True)

    # ============ PML PARAMETERS ============

    # Infinite model (True: Infinite model, False: PML scheme)
    get_ref_model = False

    # Reference frequency
    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    # Modal solver for fundamental frequency
    modal_solver = 'KRYLOVSCH_CH'

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    fr_files = max(int(100 * max(dt_usu_lst) / dt_usu), 1)
    dictionary = wave_dict_2d(dt_usu, fr_files, "source", get_ref_model)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_pml(dictionary, edge_length, f_est, 2)

    # ============ REFERENCE MODEL ============

    # Create the infinite model and get the reference signal
    if get_ref_model:
        run_reference(Wave_obj, max_divisor_tf=max_div_tf)

    # ============ PML SCHEME ============

    # Run the PML scheme
    run_pml(Wave_obj, habc_reference_freq_lst, modal_solver)


def test_loop_pml_3d():
    '''
    Loop for applying the PML3D to the model in Fig. 8 of Salas et al. (2022)
    '''

    # ============ SIMULATION PARAMETERS ============

    case = 0  # Integer from 0 to 1

    # Mesh size in km
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.150, 0.125]

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu_lst = [0.00050, 0.00048]  # Approximate eigenvalue

    # Eikonal degree
    degree_eikonal_lst = [2, 1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.05, 0.05]

    # Maximum divisor of the final time
    max_div_tf_lst = [8, 8]  # Approximate eigenvalue

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = dt_usu_lst[case]
    p_eik = degree_eikonal_lst[case]
    f_est = f_est_lst[case]
    max_div_tf = max_div_tf_lst[case]
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Timestep Size: {:.3f} ms".format(1e3 * dt_usu), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    print("Maximum Divisor of Final Time: {}".format(max_div_tf), flush=True)

    # ============ PML PARAMETERS ============

    # Infinite model (True: Infinite model, False: PML scheme)
    get_ref_model = False

    # Reference frequency
    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    # Modal solver for fundamental frequency
    modal_solver = 'KRYLOVSCH_CH'

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    fr_files = max(int(100 * max(dt_usu_lst) / dt_usu), 1)
    dictionary = wave_dict_3d(dt_usu, fr_files, "source", get_ref_model, p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_pml(dictionary, edge_length, f_est, 3)

    # ============ REFERENCE MODEL ============

    # Create the infinite model and get the reference signal
    if get_ref_model:
        run_reference(Wave_obj, max_divisor_tf=max_div_tf)

    # ============ PML SCHEME ============

    # Run the PML scheme
    run_pml(Wave_obj, habc_reference_freq_lst, modal_solver)


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    # test_loop_pml_2d()
    test_loop_pml_3d()


'''
=================================================================
DATA FOR 2D MODEL - Ele = T
----------------------------

*EIKONAL
eik_min = 83.333 ms
f_est   100m   62.5m     50m     25m     20m
 0.01 66.836   --/--   --/--   --/--   --/--
 0.02 73.308  83.907* 83.944* 83.812* 82.193
 0.03 77.178  85.322  85.068  84.398   --/--
 0.04 79.680  86.352  85.933  84.901  83.434*
 0.05 81.498  87.263  86.718  85.375  83.863
 0.06 82.942* 88.130  87.470  85.837  84.250
 0.07 84.160  88.977  88.207  86.292  84.613
 0.08 85.233  89.815  88.934  86.745  84.961

=================================================================
DATA FOR 3D MODEL - Ele = T
-----------------------------

*EIKONAL
eik_min = 83.333 ms
f_est   150m    125m
 0.03 76.777  75.378
 0.04 79.409  78.301
 0.05 82.273* 82.274*
 0.06 85.347  86.409 
'''
