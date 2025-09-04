import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
from spyro.utils.cost import comp_cost


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
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 1.0,  # thickness in km - always positive
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

    # Simulate for 1.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2. Only for hypershape
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


def get_xCR_usu(Wave_obj, dat_regr_xCR, typ_xCR, n_pts):
    '''
    Get the user-defined heuristic factor for the minimum damping ratio.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    data_regr_xCR: `list`
        Data for the regression of the parameter xCR.
        Structure: [xCR, max_errIt, max_errPK, crit_opt]
        - xCR: Values of xCR used in the regression.
          The last value IS the optimal xCR
        - max_errIt: Values of the maximum integral error.
          The last value corresponds to the optimal xCR
        - max_errPK: Values of the maximum peak error.
          The last value corresponds to the optimal xCR
        - crit_opt : Criterion for the optimal heuristic factor.
          * 'err_difference' : Difference between integral and peak errors
          * 'err_integral' : Minimum integral error
          * 'err_sum' : Sum of integral and peak errors
    typ_xCR : `str`
        Type of computation for the parameter xCR.
        Options: "candidates" and "optimal"
    n_pts : `int`
        Number of candidates for the heuristic factor xCR.
        Default is 3. Must be an odd number

    Returns
    -------
    xCR_cand : `list`
        Candidates for the heuristic factor xCR based on the
        current xCR and its bounds. The candidates are sorted
        in ascending order and current xCR is not included
    xCR_opt : `float`
        Optimal xCR based on a quadratic regression and a criterion
    '''

    # Heuristic factor for the minimum damping ratio
    if typ_xCR == "candidates":

        # Determining the xCR candidates for iterations
        xCR_cand = Wave_obj.get_xCR_candidates(n_pts=n_pts)
        return xCR_cand

    elif typ_xCR == "optimal":

        # Getting an optimal xCR
        crit_opt = dat_regr_xCR[-1]  # Criterion for optimal xCR
        xCR_opt = Wave_obj.get_xCR_optimal(dat_regr_xCR, crit_opt=crit_opt)
        return xCR_opt


def habc_fig8(Wave_obj, dat_regr_xCR, xCR_usu=None, plot_comparison=True):
    '''
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022).

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    data_regr_xCR: `list`
        Data for the regression of the parameter xCR.
        Structure: [xCR, max_errIt, max_errPK, crit_opt]
        - xCR: Values of xCR used in the regression.
          The last value IS the optimal xCR
        - max_errIt: Values of the maximum integral error.
          The last value corresponds to the optimal xCR
        - max_errPK: Values of the maximum peak error.
          The last value corresponds to the optimal xCR
        - crit_opt : Criterion for the optimal heuristic factor.
          * 'err_difference' : Difference between integral and peak errors
          * 'err_integral' : Minimum integral error
          * 'err_sum' : Sum of integral and peak errors
    xCR_usu : `float`, optional
        User-defined heuristic factor for the minimum damping ratio.
        Default is None, which defines an estimated value
    plot_comparison : `bool`, optional
        If True, the solution (time and frequency) at receivers
        and the error measures are plotted. Default is True.

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_habc_case()

    # Acquiring reference signal
    Wave_obj.get_reference_signal()

    # Determining layer size
    Wave_obj.size_habc_criterion(n_root=1)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Setting the damping profile within absorbing layer
    Wave_obj.damping_layer(xCR_usu=xCR_usu)

    # Applying NRBCs on outer boundary layer
    Wave_obj.nrbc_on_boundary_layer()

    # Solving the forward problem
    Wave_obj.forward_solve()

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Collecting data for regression
    dat_regr_xCR[0].append(Wave_obj.xCR)
    dat_regr_xCR[1].append(Wave_obj.max_errIt)
    dat_regr_xCR[2].append(Wave_obj.max_errPK)

    if plot_comparison:

        # Plotting the solution at receivers and the error measures
        Wave_obj.comparison_plots(regression_xCR=True,
                                  data_regr_xCR=dat_regr_xCR)


def test_loop_habc_3d():
    '''
    Loop for applying the HABC to the 3D-Fig.8 model in Salas et al. (2022).
    '''

    case = 0  # Integer from 0 to 3

    # ============ SIMULATION PARAMETERS ============

    # Mesh size in km
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.150, 0.125, 0.100, 0.080]

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu_lst = [0.0032, 0.0024, 0.0018, 0.0016]

    # Eikonal degree
    degree_eikonal_lst = [2, 1, 2, 1]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.04, 0.05, 0.03, 0.05]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    dt_usu = dt_usu_lst[case]
    p_eik = degree_eikonal_lst[case]
    f_est = f_est_lst[case]
    print("\nMesh Size: {:.3f} km".format(edge_length))
    print("Timestep Size: {:.3f} ms".format(1e3 * dt_usu))
    print("Eikonal Degree: {}".format(p_eik))
    print("Eikonal Stabilizing Factor: {:.2f}\n".format(f_est))

    # ============ HABC PARAMETERS ============

    # Infinite model (True: Infinite model, False: HABC scheme)
    get_ref_model = False

    # Loop for HABC cases
    loop_modeling = not get_ref_model

    # Hyperellipse degrees
    degree_layer_lst = [2]  # [None, 2, 3, 4, 5]

    # Reference frequency
    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    # Type of the hypereshape degree
    degree_type = "real"  # "integer"

    # Error criterion for heuristic factor xCR
    crit_opt = "err_sum"  # err_integral, err_peak

    # Number of points for regression (odd number)
    n_pts = 1

    # ============ MESH AND EIKONAL ============
    # Create dictionary with parameters for the model
    fr_files = max(int(100 * max(dt_usu_lst) / dt_usu), 1)
    dictionary = wave_dict(dt_usu, fr_files, "rectangular", None,
                           degree_type, "source", get_ref_model, p_eik)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_habc(dictionary, edge_length, f_est)

    # ============ REFERENCE MODEL ============
    if get_ref_model:
        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing reference get_reference_signal
        max_divisor_tf = 1  # 2 if case == 0 else 1
        Wave_obj.infinite_model(check_dt=True, max_divisor_tf=max_divisor_tf)

        # Set model parameters for the HABC scheme
        Wave_obj.abc_get_ref_model = False

        # Estimating computational resource usage
        comp_cost("tfin", tRef=tRef,
                  user_name=Wave_obj.path_save + "preamble/INF_")

    # ============ HABC SCHEME ============
    if loop_modeling:

        # Data to print on screen
        crit_str = "\nCriterion for Heuristic Factor ({:.0f} Points): {}"
        fref_str = "HABC Reference Frequency: {}\n"

        # Loop for different layer shapes and degrees
        for habc_reference_freq in habc_reference_freq_lst:

            # Reference frequency for sizing the hybrid absorbing layer
            Wave_obj.abc_reference_freq = habc_reference_freq
            print(crit_str.format(n_pts, crit_opt.replace("_", " ").title()))

            # Criterion for optinal heuristic factor xCR
            print(fref_str.format(habc_reference_freq.capitalize()))

            for degree_layer in degree_layer_lst:

                # Update the layer shape and its degree
                Wave_obj.abc_boundary_layer_shape = "hypershape" \
                    if degree_layer is not None else "rectangular"
                Wave_obj.abc_deg_layer = degree_layer

                # Data for regression of xCR parameter
                dat_regr_xCR = [[] for _ in range(3)]
                dat_regr_xCR.append(crit_opt)

                for itr_xCR in range(n_pts + 1):
                    try:
                        # User-defined heuristic factor x_CR
                        if itr_xCR == 0:
                            xCR_usu = None
                        elif itr_xCR == n_pts:
                            xCR_usu = xCR_opt
                        else:
                            xCR_usu = xCR_cand[itr_xCR - 1]

                        print("Iteration {} of {}".format(itr_xCR, n_pts))

                        # Reference to resource usage
                        tRef = comp_cost("tini")

                        # Run the HABC scheme
                        plot_comparison = True if itr_xCR == n_pts else False
                        habc_fig8(Wave_obj, dat_regr_xCR, xCR_usu=xCR_usu,
                                  plot_comparison=plot_comparison)

                        # Estimating computational resource usage
                        comp_cost("tfin", tRef=tRef,
                                  user_name=Wave_obj.path_case_habc)

                        # User-defined heuristic factor x_CR
                        if itr_xCR == 0:
                            xCR_cand = get_xCR_usu(
                                Wave_obj, dat_regr_xCR, "candidates", n_pts)
                        elif itr_xCR == n_pts - 1:
                            xCR_opt = get_xCR_usu(
                                Wave_obj, dat_regr_xCR, "optimal", n_pts)

                        if n_pts == 1:
                            break

                    except Exception as e:
                        print(f"Error Solving: {e}")
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

# n_hyp 150m 125m 100m  80m
# nmin   2.8  2.4  2.2  2.1
# nmax   4.7  4.7  4.7  4.7

# Computing exact eigenvalues is extremely computationally demanding!
# Maximum Timestep Size: 1.859 ms
# Selected Timestep Size: 1.600 ms
