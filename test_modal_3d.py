import firedrake as fire
import warnings
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(degree_type, habc_ref_freq, degree_eikonal):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
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
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": "rectangular",  # Options: rectangular or hypershape
        "degree_layer": None,  # Float >= 2 (hyp) or None (rec)
        "degree_type": degree_type,  # Options: real or integer
        "habc_reference_freq": habc_ref_freq,  # Options: source or boundary
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": False,
        "forward_output_filename": "output/forward/fw_output_3d.pvd",
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


def get_range_hyp(Wave_obj):
    '''
    Determine the range of the hyperellipse degree for the absorbing layer.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_habc_case()

    # Determining layer size
    Wave_obj.size_habc_criterion(n_root=1)


def modal_fig8(Wave_obj, modal_solver_lst, fitting_c):
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

    Returns
    -------
    None
    '''

    # Check hyperellipse degree
    get_range_hyp(Wave_obj)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Loop for different modal solvers
    for modal_solver in modal_solver_lst:

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


def test_loop_modal_3d():
    '''
    Loop for testing modals solvers in 3D
    '''

    # ============ SIMULATION PARAMETERS ============

    case = 0  # Integer from 0 to 3

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.150, 0.125, 0.100, 0.080]

    # Eikonal degree
    degree_eikonal_lst = [2, 1, 2, 1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.05, 0.05, 0.03, 0.05]

    # Parameters for fitting equivalent velocity regression
    fitting_c_lst = [(1.0, 1.0, 0.1, 0.1),
                     (1.0, 1.0, 0.1, 0.0),
                     (1.0, 1.0, 0.5, 0.5),
                     (1.0, 1.0, 0.5, 0.5)]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    p_eik = degree_eikonal_lst[case]
    f_est = f_est_lst[case]
    fitting_c = fitting_c_lst[case]
    print("\nMesh Size: {:.3f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Degree: {}".format(p_eik), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    print((fit_str + "{:.1f}\n").format(*fitting_c), flush=True)

    # ============ HABC PARAMETERS ============

    # Loop for HABC cases (True: Modal analysis, False: Hyperellipse degree)
    loop_modeling = True

    # Reference frequency
    habc_ref_freq = "source"  # "boundary"

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Hyperellipse degrees
    degree_layer_lst = [2.4, 3.0, 4.0, 4.4, None]

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict(degree_type, habc_ref_freq, p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est)

    # ============ MODAL ANALYSIS ============

    # Data to print on screen
    fref_str = "HABC Reference Frequency: {}"
    degr_str = "Type of the Hypereshape Degree: {}"

    # Reference frequency for sizing the hybrid absorbing layer
    print(fref_str.format(habc_ref_freq), flush=True)

    # Type of the hypereshape degree
    print(degr_str.format(degree_type), flush=True)

    if loop_modeling:

        # Modal solvers
        modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS',
                            'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                            'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

        for degree_layer in degree_layer_lst:

            # Update the layer shape and its degree
            Wave_obj.abc_boundary_layer_shape = "hypershape" \
                if degree_layer is not None else "rectangular"
            Wave_obj.abc_deg_layer = degree_layer

            try:
                # Computing the fundamental frequency
                modal_fig8(Wave_obj, modal_solver_lst, fitting_c)

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)
                break

            # Renaming the folder if degree_layer is modified
            Wave_obj.rename_folder_habc()

    else:

        # Update the reference frequency, the layer shape and its degree
        Wave_obj.abc_reference_freq = habc_ref_freq
        Wave_obj.abc_boundary_layer_shape = "hypershape"
        Wave_obj.abc_deg_layer = 2.

        # Getting the range of the hyperellipse degrees
        get_range_hyp(Wave_obj)


# Testing several modal solvers for 3D models
if __name__ == "__main__":
    test_loop_modal_3d()

# eik_min = 83.333 ms
# f_est   150m    125m    100m     80m
#  0.02  --/--   --/--  79.817  58.368
#  0.03 76.777  75.378  84.365* 71.123
#  0.04 79.409  78.301  89.437  78.665
#  0.05 82.273* 82.274* 93.810  83.901*
#  0.06 85.347  86.409  97.935  88.048

# Case  150m 125m 100m  80m
# p_eik    2    1    2    1

# n_hyp  150m 125m 100m  80m
# n_min   2.2  2.4  2.2  2.1
# n_max   4.4  4.7  4.7  4.7

# ANALYTICAL
# Case0     REC    N4.4    N4.0    N3.0    N2.4
# fnum  0.42136 0.44474 0.45401 0.48967 0.52709
# fana  0.42562 0.46018 0.46466 0.48649 0.51350
# fray  0.44942 0.48788 0.49239 0.51531 0.54333

# Case1     REC    N4.4    N4.0    N3.0    N2.4
# fnum  0.45563 0.47792 0.48672 0.52039 0.55546
# fana  0.45830 0.50053 0.50549 0.53045 0.55728
# fray  0.47918 0.52672 0.53146 0.55971 0.58428

# RAYLEIGH N2.4 dx = 150m
# n_eigfunc       2      *4       6
# freq(Hz)  0.65356 0.54333 0.53122
# texe(s)     0.799  34.327 373.401
# mem(MB)     6.730  47.889 154.636

# dx = 150m
# Frequency[Hz]    N2.4         (texe/pmem)    N3.0         (texe/pmem)
# ANALYTICAL    0.52709 ( 2.650s/  5.271MB) 0.48967 ( 2.300s/  2.631MB)
# ARNOLDI       0.51350 (44.335s/168.015MB) 0.48649 (62.831s/189.226MB)
# LANCZOS       0.51350 (44.562s/125.497MB) 0.48649 (63.646s/141.812MB)
# LOBPCG        0.51350 (57.811s/122.124MB) 0.48649 (62.392s/137.995MB)
# KRYLOVSCH_CH  0.51350 (11.359s/  0.106MB) 0.48649 (12.469s/  0.097MB)
# KRYLOVSCH_CG  0.51350 (10.105s/  0.103MB) 0.48649 (13.230s/  0.089MB)
# KRYLOVSCH_GH  0.51350 (10.176s/  0.099MB) 0.48649 (12.324s/  0.087MB)
# KRYLOVSCH_GG  0.51350 (10.510s/  0.103MB) 0.48649 (12.750s/  0.099MB)
# RAYLEIGH      0.54333 (34.327s/ 47.889MB) 0.51531 (33.767s/ 46.326MB)

# Frequency[Hz]    N4.0         (texe/pmem)    N4.4         (texe/pmem)
# ANALYTICAL    0.45401 ( 2.919s/  2.840MB) 0.44474 ( 2.972s/  2.853MB)
# ARNOLDI       0.46466 (73.860s/207.193MB) 0.46018 (90.358s/212.573MB)
# LANCZOS       0.46466 (74.358s/155.258MB) 0.46018 (91.671s/159.243MB)
# LOBPCG        0.46469 (82.396s/151.074MB) 0.46018 (65.233s/154.963MB)
# KRYLOVSCH_CH  0.46466 (16.599s/  0.096MB) 0.46018 (17.690s/  0.089MB)
# KRYLOVSCH_CG  0.46466 (16.782s/  0.088MB) 0.46018 (17.595s/  0.100MB)
# KRYLOVSCH_GH  0.46466 (15.698s/  0.087MB) 0.46018 (17.927s/  0.122MB)
# KRYLOVSCH_GG  0.46466 (15.880s/  0.099MB) 0.46018 (18.018s/  0.091MB)
# RAYLEIGH      0.49239 (37.112s/ 51.316MB) 0.48788 (42.566s/ 51.714MB)

# Frequency[Hz]     REC         (texe/pmem)
# ANALYTICAL    0.42136 (  9.017s/  5.521MB)
# ARNOLDI       0.42562 (420.204s/435.557MB)
# LANCZOS       0.42562 (405.329s/325.842MB)
# LOBPCG        0.42562 (124.287s/317.060MB)
# KRYLOVSCH_CH  0.42562 ( 65.094s/  0.098MB)
# KRYLOVSCH_CG  0.42562 ( 64.242s/  0.088MB)
# KRYLOVSCH_GH  0.42562 ( 65.212s/  0.088MB)
# KRYLOVSCH_GG  0.42562 ( 64.883s/  0.099MB)
# RAYLEIGH      0.44942 ( 50.484s/ 98.198MB)

# dx = 125m
# Frequency[Hz]    N2.4          (texe/pmem)   N3.0          (texe/pmem)
# ANALYTICAL    0.55546 ( 4.135s/  5.220MB) 0.52039 (  3.423s/  3.290MB)
# ARNOLDI       0.55728 (94.101s/217.026MB) 0.53045 (106.747s/242.862MB)
# LANCZOS       0.55728 (91.846s/162.140MB) 0.53045 (103.531s/181.887MB)
# LOBPCG        0.55728 (68.108s/157.168MB) 0.53045 ( 91.198s/176.951MB)
# KRYLOVSCH_CH  0.55728 (18.182s/  0.109MB) 0.53045 ( 21.511s/  0.088MB)
# KRYLOVSCH_CG  0.55728 (18.240s/  0.104MB) 0.53045 ( 21.400s/  0.089MB)
# KRYLOVSCH_GH  0.55728 (18.163s/  0.121MB) 0.53045 ( 21.206s/  0.100MB)
# KRYLOVSCH_GG  0.55728 (17.872s/  0.101MB) 0.53045 ( 21.028s/  0.096MB)
# RAYLEIGH      0.58428 (38.497s/ 57.734MB) 0.55791 ( 38.408s/ 58.414MB)

# Frequency[Hz]    N4.0          (texe/pmem)    N4.4          (texe/pmem)
# ANALYTICAL    0.48672 (  4.145s/  3.510MB) 0.44792 (  4.151s/  3.613MB)
# ARNOLDI       0.50549 (174.262s/264.199MB) 0.50053 (169.816s/264.187MB)
# LANCZOS       0.50549 (171.176s/197.834MB) 0.50053 (174.952s/197.831MB)
# LOBPCG        0.50549 (104.918s/192.466MB) 0.50053 ( 97.628s/192.465MB)
# KRYLOVSCH_CH  0.50549 ( 26.097s/  0.088MB) 0.50053 ( 26.752s/  0.089MB)
# KRYLOVSCH_CG  0.50549 ( 26.346s/  0.101MB) 0.50053 ( 26.925s/  0.088MB)
# KRYLOVSCH_GH  0.50549 ( 27.303s/  0.096MB) 0.50053 ( 27.109s/  0.091MB)
# KRYLOVSCH_GG  0.50549 ( 27.282s/  0.091MB) 0.50053 ( 26.867s/  0.096MB)
# RAYLEIGH      0.53146 ( 39.049s/ 63.262MB) 0.52672 ( 39.824s/ 61.626MB)

# Frequency[Hz]     REC         (texe/pmem)
# ANALYTICAL    0.45563 ( 12.730s/  6.686MB)
# ARNOLDI       0.45830 (656.740s/540.124MB)
# LANCZOS       0.45830 (629.617/403.936MB)
# LOBPCG        0.45833 (285.246s/393.328MB)
# KRYLOVSCH_CH  0.45830 ( 97.304s/  0.089MB)
# KRYLOVSCH_CG  0.45830 ( 98.332s/  0.087MB)
# KRYLOVSCH_GH  0.45830 ( 98.328s/  0.099MB)
# KRYLOVSCH_GG  0.45830 ( 99.114s/  0.096MB)
# RAYLEIGH      0.47918 ( 59.143s/121.611MB)

# Old results for reference:
# dx = 100m  fitting_c = (1.0, 1.0, 0.5, 0.3)
# Frequency[Hz]    N2.8           (texe/pmem)
# ANALYTICAL    0.53250 (  15.450s/  9.719MB)
# ARNOLDI       0.53140 (1548.414s/538.817MB)
# LANCZOS       0.53140 (1752.195s/403.454MB)
# LOBPCG        0.53140 ( 247.431s/392.467MB)
# KRYLOVSCH_CH  0.53140 ( 352.688s/  0.106MB)
# KRYLOVSCH_CG  0.53140 ( 265.128s/  0.102MB)
# KRYLOVSCH_GH  0.53140 ( 282.246s/  0.098MB)
# KRYLOVSCH_GG  0.53140 ( 256.501s/  0.103MB)
# RAYLEIGH      0.55468 (  69.898s/128.790MB)

# dx = 80m  fitting_c = (1.0, 1.0, 0.5, 0.4)
# Frequency[Hz]    N2.8          (texe/pmem)
# ANALYTICAL    0.53625 ( 33.422s/ 13.390MB)
# ARNOLDI       ---/--- (---/---s/---/---MB)
# LANCZOS       ---/--- (---/---s/---/---MB)
# LOBPCG        0.53036 (386.984s/650.429MB)
# KRYLOVSCH_CH  0.53036 (681.595s/  0.105MB)
# KRYLOVSCH_CG  0.53036 (682.864s/  0.098MB)
# KRYLOVSCH_GH  0.53036 (701.067s/  0.104MB)
# KRYLOVSCH_GG  0.53036 (686.890s/  0.101MB)
# RAYLEIGH      0.55424 ( 97.699s/203.975MB)
