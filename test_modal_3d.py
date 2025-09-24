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

    case = 0  # Integer from 0 to 3

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.150, 0.125, 0.100, 0.080]

    # Eikonal degree
    degree_eikonal_lst = [2, 1, 2, 1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.04, 0.05, 0.03, 0.05]

    # Parameters for fitting equivalent velocity regression
    fitting_c_lst = [(1.0, 1.0, 0.2, 0.5),
                     (1.0, 1.0, 0.5, 0.5),
                     (1.0, 1.0, 0.5, 0.5),
                     (1.0, 1.0, 0.5, 0.5)]

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
    degree_layer_lst = [2.8]  # [None, 2.8, 3, 4, 4.7]

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

# n_hyp  150m 125m 100m  80m
# n_min   2.8  2.4  2.2  2.1
# n_max   4.7  4.7  4.7  4.7

# Frequency[Hz]     REC          (texe/pmem)
# ANALYTICAL    0.45565 ( 3.434s/  5.663MB)
# ARNOLDI       0.45901 (69.961s/205.148MB)
# LANCZOS       0.45901 (68.482s/152.205MB)
# LOBPCG        0.48982 (45.466s/148.411MB)
# KRYLOVSCH_CH  0.45901 (14.263s/  0.099MB)
# KRYLOVSCH_CG  0.45901 (14.103s/  0.102MB)
# KRYLOVSCH_GH  0.45901 (14.057s/  0.100MB)
# KRYLOVSCH_GG  0.45901 (13.809s/  0.097MB)
# RAYLEIGH      0.47976 (77.643s/ 60.394MB)

# RAYLEIGH dx = 150m
# n_eigfunc       2      *4       6
# freq(Hz)  0.59662 0.47976 0.47087
# texe(s)     0.842  77.643 529.358
# mem(MB)     7.719  60.394 189.725
