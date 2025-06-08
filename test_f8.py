import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
from spyro.habc.cost import comp_cost
import ipdb


def wave_dict(layer_shape, degree_layer, habc_reference_freq, get_ref_model):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary
    get_ref_model : `bool`
        If True, the infinite model is created. If False, the absorbing layer
        is created based on the model parameters.

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
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Integer >= 2. Only for hypershape
        "habc_reference_freq": habc_reference_freq,  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": get_ref_model,  # If True, the infinite model is created
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
        "acoustic_energy_filename": "output/preamble/acoustic_potential_energy",
    }

    return dictionary


def preamble_habc(dictionary, edge_length):
    '''
    Run the infinite model and the Rikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km

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
    Wave_obj.preamble_mesh_operations()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Initializing Eikonal object
    Eik_obj = eik.Eikonal(Wave_obj)

    # Finding critical points
    Wave_obj.critical_boundary_points(Eik_obj)

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
    typ_xCR : `str`
        Type of computation for the parameter xCR.
        Options: "candidates" and "optimal"

    Returns
    -------
    xCR_cand : `list`
        List of xCR candidates based on the current xCR and its bounds.
        Current xCR is not included
    xCR_opt : `float`
        Optimal xCR based on a quadratic regression and a criterion
    '''

    # Heuristic factor for the minimum damping ratio
    if typ_xCR == "candidates":

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = Wave_obj.xCR_bounds[0]

        # Estimated intial value
        xCR = Wave_obj.xCR

        # Determining the xCR candidates for iterations
        if n_pts == 3:

            # Initial search range
            xCR_min, xCR_max = Wave_obj.xCR_bounds[1]

            xCR_rang = [xCR_inf, xCR_min, xCR_max, xCR_sup]
            unique_xCR = list(dict.fromkeys(xCR_rang))
            if xCR in unique_xCR:
                unique_xCR.remove(xCR)

            if len(unique_xCR) == 1:
                xCR_int = (unique_xCR[0] + xCR) / 2
                xCR_cand = [unique_xCR[0], xCR_int]
            else:
                xCR_cand = sorted(
                    unique_xCR, key=lambda u_xCR: abs(u_xCR - xCR))[:2]

        else:
            step = 0.25 * min(abs(xCR - xCR_inf), abs(xCR_sup - xCR))
            xCR_cand = [np.clip(xCR - i * step, xCR_inf, xCR_sup)
                        for i in range(n_pts // 2, 0, -1)] + \
                [np.clip(xCR + i * step, xCR_inf, xCR_sup)
                 for i in range(1, n_pts // 2 + 1)]
        return xCR_cand

    elif typ_xCR == "optimal":

        # Getting an optimal xCR
        crit_opt = dat_regr_xCR[-1]  # Criterion for optimal xCR
        xCR_opt = Wave_obj.get_xCR_optimal(dat_regr_xCR, crit_opt=crit_opt)
        return xCR_opt


def test_habc_fig8(Wave_obj, dat_regr_xCR, xCR_usu=None, plot_comparison=True):
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
    Wave_obj.size_habc_criterion(n_root=1,
                                 layer_based_on_mesh=True)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Setting the damping profile within absorbing layer
    Wave_obj.damping_layer(xCR_usu=xCR_usu)

    # Applying NRBCs on outer boundary layer
    Wave_obj.cos_ang_HigdonBC()

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


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":

    # Mesh size
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.05]  # [0.05, 0.02, 0.01]

    degree_layer_lst = [2, 3, 4, 5]  # [None, 2, 3, 4, 5]

    habc_reference_freq_lst = ["source", "boundary"]

    get_ref_model = False

    loop_modeling = not get_ref_model

    crit_opt = "error_difference"  # "error_integral"

    n_pts = 3  # Number of points for regression (odd number)

    for edge_length in edge_length_lst:

        # ============ MESH AND EIKONAL ============
        # Create dictionary with parameters for the model
        dictionary = wave_dict("rectangular", None, "source", get_ref_model)

        # Creating mesh and performing eikonal analysis
        Wave_obj = preamble_habc(dictionary, edge_length)

        # ============ REFERENCE MODEL ============
        if get_ref_model:
            # Reference to resource usage
            tRef = comp_cost("tini")

            # Computing reference signal
            Wave_obj.infinite_model()

            # Set model parameters for the HABC scheme
            Wave_obj.abc_get_ref_model = False

            # Estimating computational resource usage
            comp_cost("tfin", tRef=tRef,
                      user_name=Wave_obj.path_save + "preamble/INF_")

        # ============ HABC SCHEME ============
        if loop_modeling:

            # Setting odd number of points for regression
            n_pts = max(3, n_pts + 1 if n_pts % 2 == 0 else n_pts)

            # Loop for different layer shapes and degrees
            for habc_reference_freq in habc_reference_freq_lst:

                # Reference frequency for sizing the hybrid absorbing layer
                Wave_obj.abc_reference_freq = habc_reference_freq

                for degree_layer in degree_layer_lst:

                    # Update the layer shape and its degree
                    Wave_obj.abc_boundary_layer_shape = "hypershape" \
                        if degree_layer is not None else "rectangular"
                    Wave_obj.abc_deg_layer = degree_layer

                    # Data for regression of xCR parameter
                    dat_regr_xCR = [[] for _ in range(3)]
                    dat_regr_xCR.append(crit_opt)

                    for itr_xCR in range(n_pts + 1):

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
                        test_habc_fig8(Wave_obj, dat_regr_xCR, xCR_usu=xCR_usu,
                                       plot_comparison=plot_comparison)

                        # Estimating computational resource usage
                        u_name = Wave_obj.path_save + Wave_obj.case_habc + "/"
                        comp_cost("tfin", tRef=tRef, user_name=u_name)

                        # User-defined heuristic factor x_CR
                        if itr_xCR == 0:
                            xCR_cand = get_xCR_usu(
                                Wave_obj, dat_regr_xCR, "candidates", n_pts)
                        elif itr_xCR == n_pts - 1:
                            xCR_opt = get_xCR_usu(
                                Wave_obj, dat_regr_xCR, "optimal", n_pts)

    # from spyro.plots.plots import plot_xCR_opt

    # data_regr_xCR = [[0.111, 0.158, 0.319, 0.637, 1.103, 1.495, 1.887, 0.955],
    #                  [0.0178, 0.0175, 0.0164, 0.0147, 0.0128, 0.0117, 0.0110, 0.0133],
    #                  [0.0060, 0.0065, 0.0079, 0.0107, 0.0145, 0.0173, 0.0201, 0.0133],
    #                  'error_difference']

    # # data_regr_xCR = [[0.111, 0.158, 0.319, 0.637, 1.103, 1.495, 1.887, 1.887],
    # #                  [0.0178, 0.0175, 0.0164, 0.0147, 0.0128, 0.0117, 0.0110, 0.011],
    # #                  [0.0060, 0.0065, 0.0079, 0.0107, 0.0145, 0.0173, 0.0201, 0.0201],
    # #                  'error_integral']
    # plot_xCR_opt(None, data_regr_xCR)

# 0.111 1.78 0.60 5.60e-08
# 0.158 1.75 0.65 5.47e-08
# 0.319 1.64 0.79 5.03e-08
# 0.637 1.47 1.07 4.33e-08
# 0.955 1.33 1.33 3.77e-08
# 1.103 1.28 1.45 3.55e-08
# 1.495 1.17 1.73 3.09e-08
# 1.887 1.10 2.01 2.75e-08
