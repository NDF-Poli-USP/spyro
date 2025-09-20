import firedrake as fire
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost


def wave_dict():
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    None

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
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)], # ToDo
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
    }

    # Simulate for 2.0 seconds.
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
        "layer_shape": "rectangular",  # Options: rectangular or hypershape
        "degree_layer": None,  # Float >= 2 (hyp) or None (rec)
        "degree_type": "real",  # Options: real or integer
        "habc_reference_freq": "source",  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
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


def modal_fig8(Wave_obj, modal_solver_lst):
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
        Wave_obj.fundamental_frequency(method=modal_solver, monitor=True)

        # Estimating computational resource usage
        name_cost = Wave_obj.path_case_habc + modal_solver + "_"
        comp_cost("tfin", tRef=tRef, user_name=name_cost)


def test_loop_modal_2d():
    '''
    Loop for applying the HABC to the model in Fig. 8 of Salas et al. (2022).
    '''

    case = 4  # Integer from 0 to 4

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.1000, 0.0625, 0.0500, 0.0250, 0.0200]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.06, 0.02, 0.02, 0.02, 0.04]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    f_est = f_est_lst[case]
    print("\nMesh Size: {:.4f} km".format(edge_length))
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est))

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2, 3, 4, 4.4, None]

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Reference frequency for sizing the hybrid absorbing layer
    fref_str = "HABC Reference Frequency: Source\n"

    # ============ MESH AND EIKONAL ============
    # Create dictionary with parameters for the model
    dictionary = wave_dict()

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_habc(dictionary, edge_length, f_est)

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    modal_solver_lst = ['ANALYTICAL']
    # modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS',
    #                     'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
    #                     'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

    for degree_layer in degree_layer_lst:

        # Update the layer shape and its degree
        Wave_obj.abc_boundary_layer_shape = "hypershape" \
            if degree_layer is not None else "rectangular"
        Wave_obj.abc_deg_layer = degree_layer

        try:
            # Computing the fundamental frequency
            modal_fig8(Wave_obj, modal_solver_lst)

        except Exception as e:
            print(f"Error Solving: {e}")
            break

        # Renaming the folder if degree_layer is modified
        Wave_obj.rename_folder_habc()


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

# n_hyp  100m  62.5m  50m  25m  20m
# n_min   2.0    2.0  2.0  2.0  2.0
# n_max   4.4    4.8  4.7  4.7  4.6

# Frequency[Hz]    N2.0      (texe/pmem)    N3.0      (texe/pmem)
# ANALYTICAL    0.51085 (0.160s/1.324MB) 0.48363 (0.194s/1.365MB)
# ARNOLDI       0.50443 (0.098s/6.650MB) 0.48266 (0.119s/7.005MB)
# LANCZOS       0.50443 (0.058s/5.967MB) 0.48266 (0.066s/6.264MB)
# LOBPCG        0.50443 (3.756s/5.984MB) 0.48266 (3.770s/6.265MB)
# KRYLOVSCH_CH  0.50443 (0.052s/0.085MB) 0.48266 (0.044s/0.086MB)
# KRYLOVSCH_CG  0.50443 (0.033s/0.076MB) 0.48266 (0.040s/0.074MB)
# KRYLOVSCH_GH  0.50443 (0.063s/0.099MB) 0.48266 (0.040s/0.087MB)
# KRYLOVSCH_GG  0.50443 (0.035s/0.093MB) 0.48266 (0.033s/0.100MB)
# RAYLEIGH      0.52785 (1.361s/3.478MB) 0.50497 (1.329s/3.639MB)

# Frequency[Hz]    N4.0      (texe/pmem)    N4.4      (texe/pmem)
# ANALYTICAL    0.44152 (0.155s/1.333MB) 0.47040 (0.166s/1.339MB)
# ARNOLDI       0.47423 (0.094s/7.120MB) 0.47270 (0.122s/7.432MB)
# LANCZOS       0.47423 (0.074s/6.399MB) 0.47270 (0.068s/6.692MB)
# LOBPCG        0.47423 (3.832s/6.381MB) 0.47270 (3.920s/6.535MB)
# KRYLOVSCH_CH  0.47423 (0.049s/0.085MB) 0.47270 (0.046s/0.085MB)
# KRYLOVSCH_CG  0.47423 (0.040s/0.076MB) 0.47270 (0.034s/0.075MB)
# KRYLOVSCH_GH  0.47423 (0.050s/0.099MB) 0.47270 (0.039s/0.098MB)
# KRYLOVSCH_GG  0.47423 (0.033s/0.087MB) 0.47270 (0.038s/0.088MB)
# RAYLEIGH      0.49647 (1.551s/3.537MB) 0.49470 (1.432s/3.573MB)

# Frequency[Hz]    REC       (texe/pmem)
# ANALYTICAL    0.42679 (0.152s/1.342MB)
# ARNOLDI       0.45539 (0.100s/7.112MB)
# LANCZOS       0.45539 (0.060s/6.352MB)
# LOBPCG        0.45539 (3.351s/6.217MB)
# KRYLOVSCH_CH  0.45539 (0.039s/0.085MB)
# KRYLOVSCH_CG  0.45539 (0.035s/0.075MB)
# KRYLOVSCH_GH  0.45539 (0.033s/0.086MB)
# KRYLOVSCH_GG  0.45539 (0.032s/0.093MB)
# RAYLEIGH      0.47634 (1.474s/3.516MB)

# RAYLEIGH
# n_eigfunc       2       4       6        8
# freq(Hz)  0.66237 0.52785 0.51705  0.51355
# texe(s)     0.192   1.390  18.209   35.027
# mem(MB)     1.360   3.637  11.469   16.161

# Case0     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45539 0.47270 0.47423 0.48266 0.50443
# fana  0.39996 0.41002 0.41435 0.43245 0.47716

# Case1     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45567 0.47253 0.47463 0.48342 0.50330
# fana  0.45097 0.46232 0.46719 0.48760 0.53802

# Case2     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45576 0.47185 0.47390 0.48232 0.50292
# fana  0.34450 0.35409 0.35783 0.37346 0.41207

# Case3     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.46763 0.48306 0.48550 0.49408 0.51332
# fana  0.36011 0.36913 0.37302 0.38930 0.42966

# Case4     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.47498 0.49076 0.49301 0.50200 0.52376
# fana  0.38264 0.39221 0.39633 0.41362 0.45658


# from time import perf_counter  # For runtime
# tRef = perf_counter()
# print(f"Time: {perf_counter() - tRef:.4f} seconds")
