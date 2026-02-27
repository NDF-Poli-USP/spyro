import pytest
import warnings
import numpy as np
import firedrake as fire
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict_2d(layer_shape, degree_layer, degree_type, habc_ref_freq):
    '''
    Create a dictionary with parameters for the 2D model

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
    dictionary["visualization"] = {}

    return dictionary


def wave_dict_3d(layer_shape, degree_layer, degree_type,
                 habc_ref_freq, degree_eikonal):
    '''
    Create a dictionary with parameters for the 3D model

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
    degree_eikonal : `int`
        Finite element order for the Eikonal equation. Should be 1 or 2.

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
    # 1 x 1 x 1 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x, +-y sides) of the domain.
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
        "source_locations": [(-0.5, 0.25, 0.5)],  # (0.5*Lz, 0.25*Lx, 0.25*Ly)
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                               (0., 0., 0), (0., Lx, 0.),
                               (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1.5 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 1.5,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # The Ricker has an amplitude of 1.
        "output_frequency": 100,  # How frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # How frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
        "degree_type": degree_type,  # Options: real or integer
        "habc_reference_freq": habc_ref_freq,  # Options: source or boundary
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {}

    return dictionary


def preamble_modal(dictionary, edge_length, f_est, dimension):
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
    Wave_obj = habc.HABC_Wave(dictionary=dictionary,
                              output_folder=f"output/modal_test{dimension}d")

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
    for modal_solver, exp_freq in zip(modal_solver_lst, exp_val_lst):

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

        met_str = f"Fundamental Frequency {Wave_obj.dimension}D. "
        met_str += f"Method {modal_solver}"
        cmp_str = f"Expected {exp_freq:.5f}, got = {Wave_obj.fundam_freq:.5f}"
        assert np.isclose(Wave_obj.fundam_freq / exp_freq, 1., atol=1e-2), \
            "❌ " + met_str + "  → " + cmp_str
        print("✅ " + met_str + " Verified: " + cmp_str, flush=True)


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
    dictionary = wave_dict_2d("rectangular", None, "real", "source")

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est, 2)

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


@pytest.mark.slow
def test_loop_modal_3d():
    '''
    Loop for testing modals solvers in 3D
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.15

    # Eikonal degree
    p_eik = 2

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.05

    # Parameters for fitting equivalent velocity regression
    fitting_c = (1.0, 1.0, 0.1, 0.1)

    # Get simulation parameters
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.4, None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_3d("rectangular", None, "real", "source", p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est, 3)

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS',
                        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                        'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

    # Expected values
    expect_hypershape = [0.52709,
                         0.51355,
                         0.51355,
                         0.51355,
                         0.51355,
                         0.51355,
                         0.51355,
                         0.51355,
                         0.54617]

    expect_rectangular = [0.42136,
                          0.42562,
                          0.42562,
                          0.42562,
                          0.42562,
                          0.42562,
                          0.42562,
                          0.42562,
                          0.44942]

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


'''
DATA FOR 2D MODEL Δx = 100m
---------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.01  66.836
 0.02  73.308
 0.03  77.178
 0.04  79.680
 0.05  81.498
 0.06  82.942*
 0.07  84.160
 0.08  85.233

*RESULTS
Frequency[Hz]    N2.0      (texe/pmem)     REC      (texe/pmem)
ANALYTICAL    0.50807 (0.273s/2.205MB) 0.45503 (0.111s/0.525MB)
ARNOLDI       0.50443 (0.109s/6.685MB) 0.45539 (0.057s/6.780MB)
LANCZOS       0.50443 (0.064s/5.967MB) 0.45539 (0.050s/6.364MB)
LOBPCG        0.50443 (3.946s/6.009MB) 0.45539 (3.219s/6.177MB)
KRYLOVSCH_CH  0.50443 (0.067s/0.085MB) 0.45539 (0.053s/0.085MB)
KRYLOVSCH_CG  0.50443 (0.053s/0.076MB) 0.45539 (0.043s/0.075MB)
KRYLOVSCH_GH  0.50443 (0.048s/0.086MB) 0.45539 (0.042s/0.085MB)
KRYLOVSCH_GG  0.50443 (0.048s/0.100MB) 0.45539 (0.044s/0.107MB)
RAYLEIGH      0.52785 (1.956s/3.792MB) 0.47634 (1.162s/1.926MB)

*ANALYTICAL
   Case0     REC*    N4.4    N4.0    N3.0   N2.0*
fnum[Hz]  0.45539 0.47270 0.47423 0.48266 0.50443
fana[Hz]  0.45503 0.46622 0.46541 0.47813 0.50807
fray[Hz]  0.47634 0.49470 0.49647 0.50497 0.52785

*RAYLEIGH N2.0
n_eigfunc       2      *4       6       8
freq[Hz]  0.66237 0.52785 0.51705 0.51355
texe[s]     0.263   1.956   5.947  17.152
mem[MB]     1.359   3.792   8.075  13.311

DATA FOR 3D MODEL Δx = 150m
---------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.03 76.777
 0.04 79.409
 0.05 82.273*
 0.06 85.347

*RESULTS
Frequency[Hz]    N2.4         (texe/pmem)     REC          (texe/pmem)
ANALYTICAL    0.52709 ( 2.650s/  5.271MB) 0.42136 (  9.017s/  5.521MB)
ARNOLDI       0.51350 (44.335s/168.015MB) 0.42562 (420.204s/435.557MB)
LANCZOS       0.51355 (44.562s/125.497MB) 0.42562 (405.329s/325.842MB)
LOBPCG        0.51355 (57.811s/122.124MB) 0.42562 (124.287s/317.060MB)
KRYLOVSCH_CH  0.51355 (11.359s/  0.106MB) 0.42562 ( 65.094s/  0.098MB)
KRYLOVSCH_CG  0.51355 (10.105s/  0.103MB) 0.42562 ( 64.242s/  0.088MB)
KRYLOVSCH_GH  0.51355 (10.176s/  0.099MB) 0.42562 ( 65.212s/  0.088MB)
KRYLOVSCH_GG  0.51355 (10.510s/  0.103MB) 0.42562 ( 64.883s/  0.099MB)
RAYLEIGH      0.54617 (34.327s/ 47.889MB) 0.44942 ( 50.484s/ 98.198MB)

ANALYTICAL
   Case0     REC*    N4.4    N4.0    N3.0   N2.4*
fnum[Hz]  0.42136 0.44474 0.45401 0.48967 0.52709
fana[Hz]  0.42562 0.46018 0.46466 0.48649 0.51350
fray[Hz]  0.44942 0.48788 0.49239 0.51531 0.54333

RAYLEIGH N2.4
n_eigfunc       2      *4       6
freq[Hz]  0.65356 0.54333 0.53122
texe[s]     0.799  34.327 373.401
mem[MB]     6.730  47.889 154.636
'''
