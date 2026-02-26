import pytest
import warnings
import firedrake as fire
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(layer_shape, degree_layer, degree_type, habc_ref_freq):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    degree_type : `str`
        Type of the hypereshape degree. Options: 'real' or 'integer'
    habc_ref_freq : `str`
        Reference frequency for the layer size. Options: 'source' or 'boundary'

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",  # "Q",
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
    # 1 x 1 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    Lz, Lx, Ly = [1., 1., 0.]
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
        "source_locations": [(-0.5, 0.25)],  # (0.5 * Lz, 0.25 * Lx)
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
    }

    # Simulate for 2. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
        "degree_type": degree_type,  # Options: real or integer
        "habc_reference_freq": habc_ref_freq,  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": False,
        "forward_output_filename": "output/forward/fw_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": False,  # Activate energy calculation
        "acoustic_energy_filename": "output/preamble/acoustic_pot_energy",
    }

    return dictionary


def preamble_modal(dictionary, edge_length, f_est):
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
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

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


def get_range_hyp(Wave_obj, n_root=1):
    '''
    Determine the range of the hyperellipse degree for the absorbing layer.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    n_root : `int`, optional
        n-th Root selected as the size of the absorbing layer. Default is 1

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_habc_case()

    # Determining layer size
    Wave_obj.size_habc_criterion(n_root=n_root)


def modal_fig8(Wave_obj, modal_solver_lst, fitting_c, exp_val_lst, n_root=1):
    '''
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022).

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'
    fitting_c : `tuple
        Parameters for fitting equivalent velocity regression.
        Structure: (fc1, fc2, fp1, fp2):
        - fc1: Magnitude order
        - fc2: Monotonicity
        - fp1: Rectangle frequency
        - fp2: Ellipse frequency
    exp_val_lst : `list`
        List of expected values for the fundamental frequency for each solver
    n_root : `int`, optional
        n-th Root selected as the size of the absorbing layer. Default is 1

    Returns
    -------
    None
    '''

    # Check hyperellipse degree
    get_range_hyp(Wave_obj, n_root=n_root)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Loop for different modal solvers
    for modal_solver, expected_freq in zip(modal_solver_lst, exp_val_lst):

        # Modal solver
        print("\nModal Solver: {}".format(modal_solver), flush=True)

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing fundamental frequency
        Wave_obj.fundamental_frequency(
            method=modal_solver, monitor=True, fitting_c=fitting_c)

        # Estimating computational resource usage
        name_cost = Wave_obj.path_case_habc + modal_solver + "_"
        comp_cost("tfin", tRef=tRef, user_name=name_cost)

        cmp_str = f"Expected {expected_freq:.5f}, got = {self.fundam_freq:.5f}"
        assert isclose(self.fundam_freq / expected_freq, 1., atol=5e-3), \
            "❌ Fundamental Frequency 2D → " + cmp_str
        print("✅ Fundamental Frequency 2D Verified: " + cmp_str, flush=True)


def test_loop_modal_2d():
    '''
    Loop for testing modals solvers in 2D
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.06

    # Parameters for fitting equivalent velocity regression
    fitting_c = (2.0, 1.8, 1.1, 0.4)

    # Get simulation parameters
    print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.0, None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict("rectangular", None, "real", "source")

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est)

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS',
                        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                        'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

    # Expected values
    expect_hypershape = [0.50807,
                         0.50443,
                         0.50443,
                         0.50443,
                         0.50443,
                         0.50443,
                         0.50443,
                         0.50443,
                         0.52785]

    expect_rectangular = [0.45503,
                          0.45539,
                          0.45539,
                          0.45539,
                          0.45539,
                          0.45539,
                          0.45539,
                          0.45539,
                          0.47634]

    expect_values_lst = [expect_hypershape, expect_rectangular]

    for degree_layer, exp_val_lst in zip(degree_layer_lst, expect_values_lst):

        # Update the layer shape and its degree
        Wave_obj.abc_boundary_layer_shape = "hypershape" \
            if degree_layer is not None else "rectangular"
        Wave_obj.abc_deg_layer = degree_layer

        try:
            # Computing the fundamental frequency
            modal_fig8(Wave_obj, modal_solver_lst, fitting_c, exp_val_lst)

            # Renaming the folder if degree_layer is modified
            Wave_obj.rename_folder_habc()

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Modal 2D raised an exception: {str(e)}")


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_loop_modal_2d()

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

# n_hyp  100m 62.5m  50m  25m  20m
# n_min   2.0   2.0  2.0  2.0  2.0
# n_max   4.4   4.4  4.4  4.6  4.6

# ANALYTICAL
# Case0     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45539 0.47270 0.47423 0.48266 0.50443
# fana  0.45503 0.46622 0.46541 0.47813 0.50807
# fray  0.47634 0.49470 0.49647 0.50497 0.52785

# Case1     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45567 0.47253 0.47463 0.48342 0.50330
# fana  0.45183 0.45920 0.46234 0.47524 0.50658
# fray  0.47643 0.49468 0.49680 0.50570 0.52676

# Case2     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45576 0.47185 0.47390 0.48232 0.50292
# fana  0.45510 0.46558 0.47201 0.47757 0.51415
# fray  0.47686 0.49400 0.49612 0.50464 0.52642

# Case3     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.46763 0.48306 0.48550 0.49408 0.51332
# fana  0.46534 0.47554 0.48700 0.48886 0.51947
# fray  0.48753 0.50408 0.50656 0.51528 0.53773

# Case4     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.47498 0.49076 0.49301 0.50200 0.52376
# fana  0.47872 0.48446 0.48689 0.49671 0.51885
# fray  0.49448 0.51134 0.51362 0.52277 0.54570

# RAYLEIGH N2.0 dx = 100m
# n_eigfunc       2      *4       6       8
# freq(Hz)  0.66237 0.52785 0.51705 0.51355
# texe(s)     0.263   1.956   5.947  17.152
# mem(MB)     1.359   3.792   8.075  13.311

# Preliminary computational costs
# dx = 100m
# Frequency[Hz]    N2.0      (texe/pmem)    N3.0      (texe/pmem)
# ANALYTICAL    0.50807 (0.273s/2.205MB) 0.47813 (0.123s/0.588MB)
# ARNOLDI       0.50443 (0.109s/6.685MB) 0.48266 (0.105s/6.674MB)
# LANCZOS       0.50443 (0.064s/5.967MB) 0.48266 (0.075s/6.297MB)
# LOBPCG        0.50443 (3.946s/6.009MB) 0.48266 (3.769s/6.122MB)
# KRYLOVSCH_CH  0.50443 (0.067s/0.085MB) 0.48266 (0.049s/0.083MB)
# KRYLOVSCH_CG  0.50443 (0.053s/0.076MB) 0.48266 (0.043s/0.075MB)
# KRYLOVSCH_GH  0.50443 (0.048s/0.086MB) 0.48266 (0.041s/0.084MB)
# KRYLOVSCH_GG  0.50443 (0.048s/0.100MB) 0.48266 (0.042s/0.072MB)
# RAYLEIGH      0.52785 (1.956s/3.792MB) 0.50497 (1.192s/1.712MB)

# Frequency[Hz]    N4.0      (texe/pmem)    N4.4      (texe/pmem)
# ANALYTICAL    0.46541 (0.113s/0.577MB) 0.46622 (0.095s/0.527MB)
# ARNOLDI       0.47423 (0.087s/6.818MB) 0.47270 (0.078s/7.120MB)
# LANCZOS       0.47423 (0.070s/6.404MB) 0.47270 (0.058s/6.683MB)
# LOBPCG        0.47423 (3.995s/6.249MB) 0.47270 (3.680s/6.531MB)
# KRYLOVSCH_CH  0.47423 (0.041s/0.084MB) 0.47270 (0.054s/0.105MB)
# KRYLOVSCH_CG  0.47423 (0.041s/0.076MB) 0.47270 (0.041s/0.101MB)
# KRYLOVSCH_GH  0.47423 (0.043s/0.084MB) 0.47270 (0.048s/0.075MB)
# KRYLOVSCH_GG  0.47423 (0.044s/0.083MB) 0.47270 (0.052s/0.083MB)
# RAYLEIGH      0.49647 (1.010s/1.743MB) 0.49470 (1.039s/1.769MB)

# Frequency[Hz]    REC       (texe/pmem)
# ANALYTICAL    0.45503 (0.111s/0.525MB)
# ARNOLDI       0.45539 (0.057s/6.780MB)
# LANCZOS       0.45539 (0.050s/6.364MB)
# LOBPCG        0.45539 (3.219s/6.177MB)
# KRYLOVSCH_CH  0.45539 (0.053s/0.085MB)
# KRYLOVSCH_CG  0.45539 (0.043s/0.075MB)
# KRYLOVSCH_GH  0.45539 (0.042s/0.085MB)
# KRYLOVSCH_GG  0.45539 (0.044s/0.107MB)
# RAYLEIGH      0.47634 (1.162s/1.926MB)

# dx = 20m
# Frequency[Hz]    N2.0          (texe/pmem)    N3.0          (texe/pmem)
# ANALYTICAL    0.51885 (  1.503s/  5.719MB) 0.49671 (  1.069s/  4.580MB)
# ARNOLDI       0.52736 (  3.885s/143.576MB) 0.50200 (  3.504s/154.656MB)
# LANCZOS       0.52736 (  3.502s/133.458MB) 0.50200 (  3.223s/144.103MB)
# LOBPCG        0.52402 (120.227s/127.459MB) 0.65813 (127.574s/137.452MB)
# KRYLOVSCH_CH  0.52736 (  1.175s/  0.087MB) 0.50200 (  1.377s/  0.084MB)
# KRYLOVSCH_CG  0.52736 (  1.263s/  0.089MB) 0.50200 (  1.369s/  0.072MB)
# KRYLOVSCH_GH  0.52736 (  1.206s/  0.085MB) 0.50200 (  1.329s/  0.082MB)
# KRYLOVSCH_GG  0.52736 (  1.179s/  0.086MB) 0.50200 (  1.269s/  0.082MB)
# RAYLEIGH      0.54570 (  2.059s/ 20.213MB) 0.52277 (  1.755s/ 19.665MB)

# Frequency[Hz]    N4.0          (texe/pmem)    N4.4          (texe/pmem)
# ANALYTICAL    0.48689 (  1.031s/  4.532MB) 0.48446 (  1.062s/  4.851MB)
# ARNOLDI       0.49301 (  3.529s/157.880MB) 0.49076 (  4.098s/164.796MB)
# LANCZOS       0.49301 (  3.255s/147.112MB) 0.49076 (  3.745s/153.552MB)
# LOBPCG        0.49604 (128.607s/140.312MB) 0.55850 (134.292s/146.474MB)
# KRYLOVSCH_CH  0.49301 (  1.282s/  0.099MB) 0.49076 (  1.369s/  0.091MB)
# KRYLOVSCH_CG  0.49301 (  1.301s/  0.084MB) 0.49076 (  1.456s/  0.083MB)
# KRYLOVSCH_GH  0.49301 (  1.397s/  0.073MB) 0.49076 (  1.419s/  0.073MB)
# KRYLOVSCH_GG  0.49301 (  1.305s/  0.084MB) 0.49076 (  1.400s/  0.076MB)
# RAYLEIGH      0.51362 (  1.859s/ 20.312MB) 0.51134 (  1.668s/ 20.864MB)

# Frequency[Hz]     REC          (texe/pmem)
# ANALYTICAL    0.47872 (  1.102s/  4.504MB)
# ARNOLDI       0.47498 (  3.304s/156.040MB)
# LANCZOS       0.47498 (  3.167s/145.407MB)
# LOBPCG        0.48982 (130.674s/138.645MB)
# KRYLOVSCH_CH  0.47498 (  1.414s/  0.083MB)
# KRYLOVSCH_CG  0.47498 (  1.267s/  0.073MB)
# KRYLOVSCH_GH  0.47498 (  1.321s/  0.084MB)
# KRYLOVSCH_GG  0.47498 (  1.314s/  0.082MB)
# RAYLEIGH      0.49448 (  1.714s/ 19.793MB)
