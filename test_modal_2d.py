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


def test_loop_modal_2d():
    '''
    Loop for testing modals solvers in 2D
    '''

    # ============ SIMULATION PARAMETERS ============

    case = 0  # Integer from 0 to 4

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.1000, 0.0625, 0.0500, 0.0250, 0.0200]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.06, 0.02, 0.02, 0.02, 0.04]

    # Parameters for fitting equivalent velocity regression
    fitting_c_lst = [(2.0, 1.8, 1.1, 0.4),
                     (2.0, 2.0, 0.4, 0.1),
                     (1.0, 0.7, 0.9, 0.3),
                     (1.0, 1.0, 1.1, 0.4),
                     (1.0, 1.0, 0.9, 0.2)]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    f_est = f_est_lst[case]
    fitting_c = fitting_c_lst[case]
    print("\nMesh Size: {:.4f} km".format(edge_length))
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est))
    fit_str = "Fitting Parameters for Analytical Solver: "
    print((fit_str + "{:.1f}, {:.1f}, {:.1f}, {:.1f}").format(*fitting_c))

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer_lst = [2.0, 3.0, 4.0, 4.4, None]

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Reference frequency for sizing the hybrid absorbing layer
    fref_str = "HABC Reference Frequency: Source\n"

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict()

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

# ANALYTICAL
# Case0     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45539 0.47270 0.47423 0.48266 0.50443
# fana  0.45503 0.46622 0.46541 0.47813 0.50807

# Case1     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45567 0.47253 0.47463 0.48342 0.50330
# fana  0.45183 0.45920 0.46234 0.47524 0.50658

# Case2     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.45576 0.47185 0.47390 0.48232 0.50292
# fana  0.45510 0.46558 0.47201 0.47757 0.51415

# Case3     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.46763 0.48306 0.48550 0.49408 0.51332
# fana  0.46534 0.47554 0.48700 0.48886 0.51947

# Case4     REC    N4.4    N4.0    N3.0    N2.0
# fnum  0.47498 0.49076 0.49301 0.50200 0.52376
# fana  0.47872 0.48446 0.48689 0.49671 0.51885

# RAYLEIGH dx = 100m
# n_eigfunc       2      *4       6       8
# freq(Hz)  0.66237 0.52785 0.51705 0.51355
# texe(s)     0.192   1.390  18.209  35.027
# mem(MB)     1.360   3.637  11.469  16.161


# dx = 100m
# Frequency[Hz]    N2.0      (texe/pmem)    N3.0      (texe/pmem)
# ANALYTICAL    0.50807 (0.276s/2.020MB) 0.47813 (0.196s/0.569MB)
# ARNOLDI       0.50443 (0.098s/6.650MB) 0.48266 (0.119s/7.005MB)
# LANCZOS       0.50443 (0.058s/5.967MB) 0.48266 (0.066s/6.264MB)
# LOBPCG        0.50443 (3.756s/5.984MB) 0.48266 (3.770s/6.265MB)
# KRYLOVSCH_CH  0.50443 (0.052s/0.085MB) 0.48266 (0.044s/0.086MB)
# KRYLOVSCH_CG  0.50443 (0.033s/0.076MB) 0.48266 (0.040s/0.074MB)
# KRYLOVSCH_GH  0.50443 (0.063s/0.099MB) 0.48266 (0.040s/0.087MB)
# KRYLOVSCH_GG  0.50443 (0.035s/0.093MB) 0.48266 (0.033s/0.100MB)
# RAYLEIGH      0.52785 (1.361s/3.478MB) 0.50497 (1.329s/3.639MB)

# Frequency[Hz]    N4.0      (texe/pmem)    N4.4      (texe/pmem)
# ANALYTICAL    0.46541 (0.094s/0.594MB) 0.46622 (0.114s/0.595MB)
# ARNOLDI       0.47423 (0.094s/7.120MB) 0.47270 (0.122s/7.432MB)
# LANCZOS       0.47423 (0.074s/6.399MB) 0.47270 (0.068s/6.692MB)
# LOBPCG        0.47423 (3.832s/6.381MB) 0.47270 (3.920s/6.535MB)
# KRYLOVSCH_CH  0.47423 (0.049s/0.085MB) 0.47270 (0.046s/0.085MB)
# KRYLOVSCH_CG  0.47423 (0.040s/0.076MB) 0.47270 (0.034s/0.075MB)
# KRYLOVSCH_GH  0.47423 (0.050s/0.099MB) 0.47270 (0.039s/0.098MB)
# KRYLOVSCH_GG  0.47423 (0.033s/0.087MB) 0.47270 (0.038s/0.088MB)
# RAYLEIGH      0.49647 (1.551s/3.537MB) 0.49470 (1.432s/3.573MB)

# Frequency[Hz]    REC       (texe/pmem)
# ANALYTICAL    0.45503 (0.103s/0.545MB)
# ARNOLDI       0.45539 (0.100s/7.112MB)
# LANCZOS       0.45539 (0.060s/6.352MB)
# LOBPCG        0.45539 (3.351s/6.217MB)
# KRYLOVSCH_CH  0.45539 (0.039s/0.085MB)
# KRYLOVSCH_CG  0.45539 (0.035s/0.075MB)
# KRYLOVSCH_GH  0.45539 (0.033s/0.086MB)
# KRYLOVSCH_GG  0.45539 (0.032s/0.093MB)
# RAYLEIGH      0.47634 (1.474s/3.516MB)

# dx = 20m
# Frequency[Hz]    N2.0          (texe/pmem)    N3.0          (texe/pmem)
# ANALYTICAL    0.51885 (  1.239s/  5.744MB) 0.49671 (  1.353s/  5.870MB)
# ARNOLDI       0.52736 (  3.433s/143.553MB) 0.50200 (  3.673s/154.990MB)
# LANCZOS       0.52736 (  3.124s/133.459MB) 0.50200 (  3.279s/144.100MB)
# LOBPCG        0.52402 (116.373s/127.448MB) 0.65813 (125.569s/137.582MB)
# KRYLOVSCH_CH  0.52736 (  1.031s/  0.086MB) 0.50200 (  1.265s/  0.088MB)
# KRYLOVSCH_CG  0.52736 (  1.086s/  0.088MB) 0.50200 (  1.269s/  0.088MB)
# KRYLOVSCH_GH  0.52736 (  1.115s/  0.092MB) 0.50200 (  1.242s/  0.086MB)
# KRYLOVSCH_GG  0.52736 (  1.053s/  0.077MB) 0.50200 (  1.170s/  0.086MB)
# RAYLEIGH      0.54570 (  2.211s/ 20.421MB) 0.52277 (  2.073s/ 21.655MB)

# Frequency[Hz]    N4.0          (texe/pmem)    N4.4          (texe/pmem)
# ANALYTICAL    0.48689 (  1.453s/  6.110MB) 0.48446 (  1.469s/  6.286MB)
# ARNOLDI       0.49301 (  3.728s/158.246MB) 0.49076 (  3.962s/165.138MB)
# LANCZOS       0.49301 (  3.860s/147.120MB) 0.49076 (  3.650s/153.562MB)
# LOBPCG        0.49604 (126.570s/140.415MB) 0.55850 (133.468s/146.563MB)
# KRYLOVSCH_CH  0.49301 (  1.208s/  0.097MB) 0.49076 (  1.333s/  0.092MB)
# KRYLOVSCH_CG  0.49301 (  1.174s/  0.077MB) 0.49076 (  1.165s/  0.076MB)
# KRYLOVSCH_GH  0.49301 (  1.190s/  0.086MB) 0.49076 (  1.241s/  0.085MB)
# KRYLOVSCH_GG  0.49301 (  1.231s/  0.087MB) 0.49076 (  1.210s/  0.087MB)
# RAYLEIGH      0.51362 (  1.938s/ 22.003MB) 0.51134 (  2.045s/ 22.884MB)

# Frequency[Hz]     REC          (texe/pmem)
# ANALYTICAL    0.47872 (  1.264s/  6.090MB)
# ARNOLDI       0.47498 (  3.464s/156.369MB)
# LANCZOS       0.47498 (  3.351s/145.416MB)
# LOBPCG        0.48982 (125.923s/138.745MB)
# KRYLOVSCH_CH  0.47498 (  1.127s/  0.088MB)
# KRYLOVSCH_CG  0.47498 (  1.093s/  0.077MB)
# KRYLOVSCH_GH  0.47498 (  1.130s/  0.086MB)
# KRYLOVSCH_GG  0.47498 (  1.096s/  0.086MB)
# RAYLEIGH      0.49448 (  2.101s/ 21.516MB)
