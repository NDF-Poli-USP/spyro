import firedrake as fire
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost


def wave_dict(degree_eikonal):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
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
        "source_locations": [(-1. / 2., 1. / 4., 1. / 2.)],
        "frequency": 5.0,  # in Hz
        "delay": 1. / 3.,
        "delay_type": "time",  # "multiples_of_minimun" or "time"
        "receiver_locations": [(-1., 0., 0.), (-1., 1., 0.),
                               (0., 1., 0.), (0., 0., 0),
                               (-1., 0., 1.), (-1., 1., 1.),
                               (0., 1., 1.), (0., 0., 1.)]
    }

    # Simulate for 1.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
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
        "degree_type": "real",  # Options: real or integer
        "habc_reference_freq": "source",  # Options: source or boundary
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
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
        Structure: (fc1, fc2, fp1, fp2).

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

    # Loop for different modal solvers
    for modal_solver in modal_solver_lst:

        # Modal solver
        print("\nModal Solver: {}".format(modal_solver))

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

    case = 1  # Integer from 0 to 1

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.150, 0.125]

    # Eikonal degree
    degree_eikonal_lst = [2, 1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.04, 0.05]

    # Parameters for fitting equivalent velocity regression
    fitting_c_lst = [(1.0, 1.0, 0.1, 0.1),
                     (1.0, 1.0, 0.1, 0.0)]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    p_eik = degree_eikonal_lst[case]
    f_est = f_est_lst[case]
    fitting_c = fitting_c_lst[case]
    print("\nMesh Size: {:.4f} km".format(edge_length))
    print("Eikonal Degree: {}".format(p_eik))
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est))
    fit_str = "Fitting Parameters for Analytical Solver: "
    print((fit_str + "{:.1f}, {:.1f}, {:.1f}, {:.1f}").format(*fitting_c))

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.8, 3.0, 4.0, 4.7, None]

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Reference frequency for sizing the hybrid absorbing layer
    fref_str = "HABC Reference Frequency: Source\n"

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict(p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_modal(dictionary, edge_length, f_est)

    # ============ MODAL ANALYSIS ============

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

        except Exception as e:
            print(f"Error Solving: {e}")
            break

        # Renaming the folder if degree_layer is modified
        Wave_obj.rename_folder_habc()


# Testing several modal solvers for 3D models
if __name__ == "__main__":
    test_loop_modal_3d()

# eik_min = 83.333 ms
# f_est   150m    125m    100m     80m
#  0.02  --/--   --/--  79.817  58.368
#  0.03 78.201  75.378  84.365* 71.123
#  0.04 84.857* 78.301  89.437  78.665
#  0.05 91.477  82.274* 93.810  83.901*
#  0.06 97.574  86.409  97.935  88.048

# Case  150m 125m 100m  80m
# p_eik    2    1    2    1

# n_hyp  150m 125m 100m  80m
# n_min   2.8  2.4  2.2  2.1
# n_max   4.7  4.7  4.7  4.7

# ANALYTICAL
# Case0     REC    N4.7    N4.0    N3.0    N2.8
# fnum  0.45901 0.49643 0.50407 0.52873 0.53411
# fana  0.45706 0.47593 0.49208 0.53034 0.54167

# Case1     REC    N4.7    N4.0    N3.0    N2.8    N2.4
# fnum  0.45830 0.49442 0.50325 0.52600 0.52931 0.54989
# fana  0.45563 0.47242 0.48672 0.52039 0.53031 0.55546

# (fc1, fc2, fp1, fp2)
# fc1: Magnitude order
# fc2: Monotonicity
# fp1: Rec frequency
# fp2: Hyp frequency

# RAYLEIGH dx = 150m
# n_eigfunc       2      *4       6
# freq(Hz)  0.59662 0.47976 0.47087
# texe(s)     0.842  77.643 529.358
# mem(MB)     7.719  60.394 189.725

# dx = 150m
# Frequency[Hz]    N2.8         (texe/pmem)    N3.0         (texe/pmem)
# ANALYTICAL    0.54167 ( 2.006s/  4.970MB) 0.53034 ( 2.384s/  4.959MB)
# ARNOLDI       0.53318 (34.939s/142.157MB) 0.52797 (38.541s/141.400MB)
# LANCZOS       0.53318 (33.521s/106.452MB) 0.52797 (37.850s/105.888MB)
# LOBPCG        0.53318 (47.434s/103.563MB) 0.52797 (48.990s/103.014MB)
# KRYLOVSCH_CH  0.53318 (12.187s/  0.107MB) 0.52797 (12.742s/  0.105MB)
# KRYLOVSCH_CG  0.53318 (12.402s/  0.103MB) 0.52797 (12.785s/  0.103MB)
# KRYLOVSCH_GH  0.53318 (12.428s/  0.097MB) 0.52797 (12.883s/  0.097MB)
# KRYLOVSCH_GG  0.53318 (12.311s/  0.103MB) 0.52797 (13.155s/  0.104MB)
# RAYLEIGH      0.55679 (32.015s/ 42.594MB) 0.55124 (32.398s/ 42.436MB)

# Frequency[Hz]    N4.0         (texe/pmem)    N4.7         (texe/pmem)
# ANALYTICAL    0.49208 ( 2.324s/  5.049MB) 0.47593 ( 2.297s/  5.159MB)
# ARNOLDI       0.50567 (46.561s/154.277MB) 0.49484 (38.273s/157.333MB)
# LANCZOS       0.50567 (47.516s/115.601MB) 0.49484 (36.836s/117.915MB)
# LOBPCG        0.50567 (58.811s/112.485MB) 0.49484 (60.099s/114.702MB)
# KRYLOVSCH_CH  0.50567 (18.181s/  0.108MB) 0.49484 (17.083s/  0.105MB)
# KRYLOVSCH_CG  0.50567 (17.935s/  0.124MB) 0.49484 (17.058s/  0.102MB)
# KRYLOVSCH_GH  0.50567 (18.103s/  0.104MB) 0.49484 (17.215s/  0.098MB)
# KRYLOVSCH_GG  0.50567 (17.912s/  0.102MB) 0.49484 (17.207s/  0.103MB)
# RAYLEIGH      0.52875 (35.620s/ 44.904MB) 0.51799 (33.441s/ 46.041MB)

# Frequency[Hz]     REC         (texe/pmem)
# ANALYTICAL    0.45706 ( 3.467s/  5.659MB)
# ARNOLDI       0.45901 (74.366s/204.118MB)
# LANCZOS       0.45901 (77.912s/152.536MB)
# LOBPCG        0.45901 (48.960s/148.409MB)
# KRYLOVSCH_CH  0.45901 (15.584s/  0.087MB)
# KRYLOVSCH_CG  0.45901 (16.180s/  0.112MB)
# KRYLOVSCH_GH  0.45901 (15.789s/  0.106MB)
# KRYLOVSCH_GG  0.45901 (15.151s/  0.104MB)
# RAYLEIGH      0.47976 (39.700s/ 55.415MB)

# dx = 125m
# Frequency[Hz]    N2.4          (texe/pmem)    N2.8          (texe/pmem)
# ANALYTICAL    0.57132 (  5.515s/  6.248MB) 0.54545 (  6.088s/  6.467MB)
# ARNOLDI       0.54989 (152.597s/270.626MB) 0.52931 (205.701s/294.170MB)
# LANCZOS       0.54989 (153.432s/202.672MB) 0.52931 (202.011s/220.315MB)
# LOBPCG        0.54989 ( 90.521s/197.140MB) 0.52931 ( 91.082s/214.372MB)
# KRYLOVSCH_CH  0.54989 ( 49.876s/  0.129MB) 0.52931 ( 63.021s/  0.104MB)
# KRYLOVSCH_CG  0.54989 ( 49.354s/  0.125MB) 0.52931 ( 63.662s/  0.106MB)
# KRYLOVSCH_GH  0.54989 ( 50.225s/  0.106MB) 0.52931 ( 63.383s/  0.102MB)
# KRYLOVSCH_GG  0.54989 ( 50.658s/  0.102MB) 0.52931 ( 63.186s/  0.108MB)
# RAYLEIGH      0.57355 ( 41.119s/ 70.123MB) 0.55254 ( 46.284s/ 75.530MB)

# Frequency[Hz]    N3.0          (texe/pmem)    N4.0          (texe/pmem)
# ANALYTICAL    0.53525 (  6.799s/  6.648MB) 0.50062 (  6.538s/  7.020MB)
# ARNOLDI       0.52600 (241.925s/303.969MB) 0.50325 (272.538s/340.046MB)
# LANCZOS       0.52600 (238.545s/227.666MB) 0.50325 (263.691s/254.721MB)
# LOBPCG        0.52600 (122.774s/221.447MB) 0.50325 (140.752s/247.833MB)
# KRYLOVSCH_CH  0.52600 ( 76.163s/  0.131MB) 0.50325 ( 81.184s/  0.103MB)
# KRYLOVSCH_CG  0.52600 ( 74.151s/  0.125MB) 0.50325 ( 81.050s/  0.106MB)
# KRYLOVSCH_GH  0.52600 ( 72.709s/  0.105MB) 0.50325 ( 81.589s/  0.102MB)
# KRYLOVSCH_GG  0.52600 ( 73.708s/  0.102MB) 0.50325 ( 81.358s/  0.108MB)
# RAYLEIGH      0.55021 ( 46.358s/ 77.413MB) 0.52619 ( 50.652s/ 85.582MB)

# Frequency[Hz]    N4.7          (texe/pmem)    REC           (texe/pmem)
# ANALYTICAL    0.48591 (  7.334s/  7.099MB) 0.46864 ( 14.197s/  9.361MB)
# ARNOLDI       0.49442 (345.013s/346.784MB) 0.45830 (802.349s/540.732MB)
# LANCZOS       0.49442 (331.531s/259.771MB) 0.45830 (741.509s/403.947MB)
# LOBPCG        0.49442 (150.927s/252.747MB) 0.45833 (283.836s/393.509MB)
# KRYLOVSCH_CH  0.49442 ( 89.600s/  0.103MB) 0.45830 ( 96.811s/  0.107MB)
# KRYLOVSCH_CG  0.49442 ( 89.805s/  0.106MB) 0.45830 ( 98.534s/  0.102MB)
# KRYLOVSCH_GH  0.49442 ( 92.358s/  0.102MB) 0.45830 ( 95.955s/  0.121MB)
# KRYLOVSCH_GG  0.49442 ( 87.125s/  0.108MB) 0.45830 ( 96.408s/  0.102MB)
# RAYLEIGH      0.51701 ( 86.167s/ 87.055MB) 0.47918 ( 65.507s/127.942MB)

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
