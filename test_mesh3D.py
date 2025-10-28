import firedrake as fire
import warnings
import spyro.habc.habc as habc
import spyro.habc.eik as eik
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(dt_usu, fr_files, layer_shape, degree_layer, degree_type,
              habc_reference_freq, get_ref_model, degree_eikonal):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation
    fr_files : `int`
        Frequency of the output files to be saved in the simulation
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    degree_type : `str`
        Type of the hypereshape degree. Options: 'real' or 'integer'
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
    dictionary["mesh"] = {
        "Lz": 1.,  # depth in km - always positive
        "Lx": 1.,  # width in km - always positive
        "Ly": 1.,  # thickness in km - always positive
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
        "delay_type": "time",  # "multiples_of_minimun" or "time"
        "receiver_locations": [(-1., 0., 0.), (-1., 1., 0.),
                               (0., 1., 0.), (0., 0., 0),
                               (-1., 0., 1.), (-1., 1., 1.),
                               (0., 1., 1.), (0., 0., 1.)]
    }

    # Simulate for 1.5 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 1.5,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
        "degree_type": degree_type,  # Options: real or integer
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
        "acoustic_energy_filename": "output/preamble/acoustic_pot_energy_3d",
    }

    return dictionary


def preamble_habc(dictionary, edge_length, f_est):
    '''
    Run the infinite model and the Rikonal analysis

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
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

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


def habc_fig8(Wave_obj, modal_solver, xCR_usu=None):
    '''
    Apply the HABC to the model 3D based on Fig. 8 of Salas et al. (2022)

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
   modal_solver : `str`
        Method to use for solving the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'
    fitting_c : `tuple
        Parameters for fitting equivalent velocity regression.
        Structure: (fc1, fc2, fp1, fp2)
    xCR_usu : `float`, optional
        User-defined heuristic factor for the minimum damping ratio.
        Default is None, which defines an estimated value

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_habc_case()

    # Determining layer size
    Wave_obj.size_habc_criterion(n_root=1)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Setting the damping profile within absorbing layer
    Wave_obj.damping_layer(xCR_usu=xCR_usu, method=modal_solver)

    # Applying NRBCs on outer boundary layer
    Wave_obj.nrbc_on_boundary_layer()

    # Solving the forward problem
    Wave_obj.forward_solve()


def test_loop_habc_3d():
    '''
    Loop for HABC in model 3D based on Fig. 8 of Salas et al. (2022)
    '''

    case = 0  # Integer from 0 to 1

    # ============ SIMULATION PARAMETERS ============

    # Mesh size in km
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.125]

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu_lst = [0.00040]

    # Eikonal degree
    degree_eikonal_lst = [1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.05]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = dt_usu_lst[case]
    p_eik = degree_eikonal_lst[case]
    f_est = f_est_lst[case]
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Timestep Size: {:.3f} ms".format(1e3 * dt_usu), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

    # ============ HABC PARAMETERS ============

    # Reference frequency
    habc_reference_freq_lst = ["source"]

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Hyperellipse degrees
    degree_layer_lst = [2.8]

    # Modal solver for fundamental frequency
    modal_solver = 'KRYLOVSCH_CH'  # 'ANALYTICAL', 'RAYLEIGH'

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    fr_files = max(int(100 * max(dt_usu_lst) / dt_usu), 1)
    dictionary = wave_dict(dt_usu, fr_files, "rectangular", None,
                           degree_type, "source", False, p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_habc(dictionary, edge_length, f_est)

    # Name of the file containing the mesh
    Wave_obj.filename_mesh = "transfinite_cube_extended_pad3.msh"

    # ============ HABC SCHEME ============

    # Data to print on screen
    fref_str = "HABC Reference Frequency: {}\n"
    degr_str = "Type of the Hypereshape Degree: {}"
    mods_str = "Modal Solver for Fundamental Frequency: {}\n"

    # Loop for different layer shapes and degrees
    for habc_ref_freq in habc_reference_freq_lst:

        # Criterion for optinal heuristic factor xCR
        Wave_obj.abc_reference_freq = habc_ref_freq
        print(fref_str.format(habc_ref_freq.capitalize()), flush=True)

        # Type of the hypereshape degree
        print(degr_str.format(degree_type), flush=True)

        # Modal solver for fundamental frequency
        print(mods_str.format(modal_solver), flush=True)

        for degree_layer in degree_layer_lst:

            # Update the layer shape and its degree
            Wave_obj.abc_boundary_layer_shape = "hypershape" \
                if degree_layer is not None else "rectangular"
            Wave_obj.abc_deg_layer = degree_layer

            try:
                # Reference to resource usage
                tRef = comp_cost("tini")

                # Run the HABC scheme
                habc_fig8(Wave_obj, modal_solver)

                # Estimating computational resource usage
                comp_cost("tfin", tRef=tRef,
                          user_name=Wave_obj.path_case_habc)

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)
                break

            # Renaming the folder if degree_layer is modified
            Wave_obj.rename_folder_habc()


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022) in 3D
if __name__ == "__main__":
    test_loop_habc_3d()

# eik_min = 83.333 ms
# f_est   150m    125m    100m     80m
#  0.02  --/--   --/--  79.817  58.368
#  0.03 78.201  75.378  84.365* 71.123
#  0.04 84.857* 78.301  89.437  78.665
#  0.05 91.477  82.274* 93.810  83.901*
#  0.06 97.574  86.409  97.935  88.048

# SOU-1st 150m 125m 100m  80m
# n_min    2.8  2.4  2.2  2.1
# n_max    4.7  4.7  4.7  4.7

# BND-1st 150m 125m
# n_min    2.0  2.0
# n_max    4.0  4.2

# SOU-2nd 150m 125m
# n_min    2.0
# n_max    4.0

# BND-2nd 150m 125m
# n_min    2.0
# n_max    3.6

# Optional models
# edge_length_lst = [0.100, 0.080]
# dt_usu_lst = [0.0018, 0.0016]
# degree_eikonal_lst = [2, 1]
# f_est_lst = [0.03, 0.05]

# H2.8
# Err(%)     1        2        3        4        5
# eI     50.19   128.36   127.04    33.81    33.81
# eP     18.60    50.81    42.68    21.69    21.69
# Ea  1.19e-05 2.16e-05 5.22e-04 1.77e-04 1.77e-04

# REC
# Err(%)     1        2        3        4        5
# eI     11.32     8.59     6.20     2.79     2.79
# eP     14.06    13.10    11.33     7.47     7.47
# Ea  1.93e-05 4.60e-05 1.03e-04 1.87e-04 1.87e-04

# H2.8
# cosHig     Hig     Som
# eI       66.74   67.19
# eP       28.22   28.22
# Ea     7.03e-6 7.18e-6


# Computing Error Measures
# Maximum Integral Error: 94.46%
# Maximum Peak Error: 66.84%
# Acoustic Energy: 1.28e-05

# NAN


# 156
