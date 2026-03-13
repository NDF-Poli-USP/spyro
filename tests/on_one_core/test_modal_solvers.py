import pytest
import warnings
import numpy as np
import firedrake as fire
import spyro.habc.habc as habc
from os import makedirs, path
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


def create_folder(folder):
    '''
    Verify if a folder exists, if not, it creates the folder

    Parameters
    ----------
    folder: `str`
        Path to the folder to be created

    Returns
    -------
    None
    '''

    # Create the folder if it does not exist
    if not path.isdir(folder):
        makedirs(folder)


def preamble_modal(dictionary, edge_length, f_est,
                   dimension, homogeneous=True):
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
    if homogeneous:
        Wave_obj.set_initial_velocity_model(constant=1.5)

    else:
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


def run_modal(Wave_obj, modal_solver_lst, fitting_c, exp_value, n_root=1):
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
    exp_value : `float`
        Expected value for the fundamental frequency
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
    for modal_solver in modal_solver_lst:

        # Modal solver
        print("\nModal Solver: {}".format(modal_solver), flush=True)

        # Create the output folder if it does not exist
        create_folder(Wave_obj.path_case_habc)

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing fundamental frequency
        Wave_obj.fundamental_frequency(
            method=modal_solver, monitor=True, fitting_c=fitting_c)

        # Estimating computational resource usage
        name_cost = Wave_obj.path_case_habc + modal_solver + "_"
        comp_cost("tfin", tRef=tRef, user_name=name_cost)

        tol = 0.07 if (modal_solver == 'ANALYTICAL'
                       or modal_solver == 'RAYLEIGH') else 0.05

        lay_str = Wave_obj.path_case_habc.split("output/")[1].rstrip("/")[:-4]
        met_str = f"Fundamental Frequency {lay_str} {Wave_obj.dimension}D. "
        met_str += f"Method {modal_solver}"
        cmp_str = f"Expected {exp_value:.5f}, got = {Wave_obj.fundam_freq:.5f}"
        assert np.isclose(Wave_obj.fundam_freq / exp_value, 1., atol=tol), \
            "❌ " + met_str + "  → " + cmp_str
        print("✅ " + met_str + " Verified: " + cmp_str, flush=True)


def loop_modal(parameters, dictionary, degree_layer_lst,
               expect_values_lst, dimension,
               homogeneous, modal_solver_lst):
    '''
    Loop for testing modals solvers.

    Parameters
    ----------
    parameters : `list`
        List containing the parameters for the model.
        Structure: [edge_length, f_est, fitting_c]
        - edge_length : `float`
            Mesh size in km
        - f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq.
        - fitting_c : `tuple`
            Parameters for fitting equivalent velocity regression
    dictionary : `dict`
        Dictionary containing the parameters for the model
    degree_layer_lst : `list`
        List of hypershape degrees for the absorbing layer
    expect_values_lst : `list`
        List of expected values for the fundamental frequency
    dimension : `int`
        Dimension of the model (2 or 3)
    homogeneous : `bool`
        If True, the velocity model is homogeneous.
        If False, it is heterogeneous.
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'

    Returns
    -------
    None
    '''

    # Model parameters
    edge_length, f_est, fitting_c = parameters

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est, dimension,
                              homogeneous=homogeneous)

    for degree_layer, exp_value in zip(degree_layer_lst, expect_values_lst):

        # Update the layer shape and its degree
        Wave_obj.abc_boundary_layer_shape = "hypershape" \
            if degree_layer is not None else "rectangular"
        Wave_obj.abc_deg_layer = degree_layer

        try:
            # Computing the fundamental frequency
            run_modal(Wave_obj, modal_solver_lst, fitting_c, exp_value)

            # Renaming the folder if degree_layer is modified
            Wave_obj.rename_folder_habc()

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Modal {dimension}D "
                        f"raised an exception: {str(e)}")


@pytest.mark.parametrize("homogeneous", [True, False])
def test_loop_modal_2d(homogeneous):
    '''
    Test of modal solvers for 2D case

    Parameters
    ----------
    homogeneous : `bool`
        If True, the velocity model is homogeneous.
        If False, it is heterogeneous.

    Returns
    -------
    None
    '''

    c_dist = "Homogeneous" if homogeneous else "Heterogeneous"
    print("\n" + 70 * "=" + "\nTesting Modal Solvers and T elements for "
          + f"2D case. Propagation Speed: {c_dist}\n" + 70 * "=", flush=True)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.06

    # Parameters for fitting equivalent velocity regression
    if homogeneous:
        fitting_c = (0.0, 0.0, 0.0, 0.0)
    else:
        fitting_c = (0.5, 0.3, -2.2, -1.3)

    # Get simulation parameters
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # Model parameters
    parameters = [edge_length, f_est, fitting_c]

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.0, None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_2d("rectangular", None, "real", "source")

    # ============ EXPECTED VALUES ============

    # Expected values
    if homogeneous:
        expect_hypershape = 0.51046
        expect_rectangular = 0.46875
    else:
        expect_hypershape = 0.50440
        expect_rectangular = 0.45539
    expect_values_lst = [expect_hypershape, expect_rectangular]

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS',
                        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                        'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

    loop_modal(parameters, dictionary, degree_layer_lst,
               expect_values_lst, 2, homogeneous, modal_solver_lst)


# @pytest.mark.slow
@pytest.mark.parametrize("homogeneous", [True, False])
def test_loop_modal_3d_with_Tele(homogeneous):
    '''
    Test of modal solvers for 3D case

    Parameters
    ----------
    homogeneous : `bool`
        If True, the velocity model is homogeneous.
        If False, it is heterogeneous.

    Returns
    -------
    None
    '''

    c_dist = "Homogeneous" if homogeneous else "Heterogeneous"
    print("\n" + 70 * "=" + "\nTesting Modal Solvers and T elements for "
          + f"3D case. Propagation Speed: {c_dist}\n" + 70 * "=", flush=True)

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
    if homogeneous:
        fitting_c = (0.0, 0.0, 0.0, 0.0)
    else:
        fitting_c = (0.4, 0.2, 0.5, -1.0)

    # Get simulation parameters
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # Model parameters
    parameters = [edge_length, f_est, fitting_c]

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.4, None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_3d("rectangular", None, "real", "source", p_eik)

    # ============ EXPECTED VALUES ============

    # Expected values
    if homogeneous:
        expect_hypershape = 0.52453
        expect_rectangular = 0.47727
    else:
        expect_hypershape = 0.51535
        expect_rectangular = 0.42562
    expect_values_lst = [expect_hypershape, expect_rectangular]

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL', 'KRYLOVSCH_CH',
                        'KRYLOVSCH_GH', 'RAYLEIGH']

    loop_modal(parameters, dictionary, degree_layer_lst,
               expect_values_lst, 3, homogeneous, modal_solver_lst)


# @pytest.mark.slow
@pytest.mark.parametrize("homogeneous", [True, False])
def test_loop_modal_3d_with_Qele(homogeneous):
    '''
    Test of modal solvers for 3D case

    Parameters
    ----------
    homogeneous : `bool`
        If True, the velocity model is homogeneous.
        If False, it is heterogeneous.

    Returns
    -------
    None
    '''

    c_dist = "Homogeneous" if homogeneous else "Heterogeneous"
    print("\n" + 70 * "=" + "\nTesting Modal Solvers for 2D case. "
          + f"Propagation Speed: {c_dist}.\nTest only the rectangular "
          + "case with Q elements as the hypershape  layer is not "
          + "supported for Q elements yet.\n" + 70 * "=", flush=True)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.15

    # Eikonal degree
    p_eik = 2

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.08

    # Parameters for fitting equivalent velocity regression
    if homogeneous:
        fitting_c = (0.0, 0.0, 0.0, 0.0)
    else:
        fitting_c = (0.3, 0.0, 0.5, -1.0)

    # Get simulation parameters
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # Model parameters
    parameters = [edge_length, f_est, fitting_c]

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_3d("rectangular", None, "real", "source", p_eik)
    dictionary["options"]["cell_type"] = "Q"

    # ============ EXPECTED VALUES ============

    # Expected values
    if homogeneous:
        expect_rectangular = 0.47727
    else:
        expect_rectangular = 0.41127
    expect_values_lst = [expect_rectangular]

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG',
                        'KRYLOVSCH_CG', 'KRYLOVSCH_GG', 'RAYLEIGH']

    loop_modal(parameters, dictionary, degree_layer_lst,
               expect_values_lst, 3, homogeneous, modal_solver_lst)


'''
=================================================================
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
Frequency[Hz]    N2.5      (texe/pmem)     REC      (texe/pmem)
ANALYTICAL    0.50428 (0.608s/3.135MB) 0.45737 (0.215s/0.745MB)
ARNOLDI       0.50440 (0.149s/6.692MB) 0.45539 (0.072s/6.775MB)
LANCZOS       0.50440 (0.073s/5.964MB) 0.45539 (0.064s/6.339MB)
LOBPCG        0.50440 (4.574s/5.886MB) 0.45539 (4.054s/6.185MB)
KRYLOVSCH_CH  0.50440 (0.047s/0.085MB) 0.45539 (0.047s/0.074MB)
KRYLOVSCH_CG  0.50440 (0.042s/0.072MB) 0.45539 (0.041s/0.099MB)
KRYLOVSCH_GH  0.50440 (0.051s/0.078MB) 0.45539 (0.039s/0.089MB)
KRYLOVSCH_GG  0.50440 (0.046s/0.091MB) 0.45539 (0.043s/0.081MB)
RAYLEIGH      0.52783 (1.803s/3.601MB) 0.47634 (1.238s/1.711MB)

*ANALYTICAL
   Case0     REC*   N2.0*
fnum[Hz]  0.45737 0.50428
fana[Hz]  0.45503 0.50807
fray[Hz]  0.47634 0.52783

*RAYLEIGH N2.0
n_eigfunc       2      *4       6       8
freq[Hz]  0.66237 0.52783 0.51705 0.51355
texe[s]     0.263   1.956   5.947  17.152
mem[MB]     1.359   3.792   8.075  13.311

=================================================================
DATA FOR 3D MODEL Δx = 150m - Ele = T
--------------------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.03 76.777
 0.04 79.409
 0.05 82.273*
 0.06 85.347

*RESULTS
Frequency[Hz]    N2.4         (texe/pmem)     REC          (texe/pmem)
ANALYTICAL    0.51833 (  5.853s/ 10.505MB) 0.42415 (  9.056s/  7.496MB)
ARNOLDI       0.51535 (148.976s/276.158MB) 0.42562 (456.837s/435.551MB)
LANCZOS       0.51535 (158.293s/206.282MB) 0.42562 (446.329s/325.837MB)
LOBPCG        0.51535 ( 94.996s/200.741MB) 0.42562 (134.146s/317.041MB)
KRYLOVSCH_CH  0.51535 ( 28.254s/  0.087MB) 0.42562 ( 70.890s/  0.068MB)
KRYLOVSCH_CG  0.51535 ( 28.122s/  0.080MB) 0.42562 ( 70.301s/  0.075MB)
KRYLOVSCH_GH  0.51535 ( 28.123s/  0.078MB) 0.42562 ( 71.568s/  0.098MB)
KRYLOVSCH_GG  0.51535 ( 28.2230s/ 0.090MB) 0.42562 ( 72.420s/  0.090MB)
RAYLEIGH      0.54617 ( 44.996s/ 72.103MB) 0.44942 ( 55.028s/100.738MB)

ANALYTICAL
   Case0     REC*  N2.4*
fnum[Hz]  0.42136 0.51833
fana[Hz]  0.42562 0.51535
fray[Hz]  0.44942 0.54617

RAYLEIGH N2.4
n_eigfunc       2      *4       6
freq[Hz]  0.65356 0.54617 0.53122
texe[s]     0.799  34.327 373.401
mem[MB]     6.730  47.889 154.636

=================================================================
DATA FOR 3D MODEL Δx = 150m - Ele = Q
--------------------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.02  69.442
 0.03  70.974
 0.04  73.179
 0.05  75.766
 0.06  78.548
 0.07  81.431
 0.08  84.377*
 0.09  87.376

*RESULTS
Frequency[Hz]     REC          (texe/pmem)
ANALYTICAL    0.41373 ( 5.035s/ 11.136MB)
ARNOLDI       0.41127 (34.776s/327.425MB)
LANCZOS       0.41127 (32.790s/218.837MB)
LOBPCG        0.41127 (36.293s/215.606MB)
KRYLOVSCH_CH  0.41127 (25.936s/  0.102MB)
KRYLOVSCH_CG  0.41127 (25.432s/  0.086MB)
KRYLOVSCH_GH  0.41127 (25.274s/  0.085MB)
KRYLOVSCH_GG  0.41127 (25.735s/  0.105MB)
RAYLEIGH      0.43304 (25.615s/ 51.299MB)

ANALYTICAL
   Case0     REC*
fnum[Hz]  0.41127
fana[Hz]  0.41373
fray[Hz]  0.43304

RAYLEIGH REC
n_eigfunc       2      *4       6
freq[Hz]  0.50637 0.43304 0.42081
texe[s]     0.859  25.615 497.458
mem[MB]     8.168  51.299 185.377
'''
