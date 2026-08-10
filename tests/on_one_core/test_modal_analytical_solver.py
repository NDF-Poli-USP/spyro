"""Unit tests for the Analytical Modal solver in spyro.solvers.modal.modal_ana_sol.

These tests verify the analytical modal solver by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from pytest import fail, fixture, mark, param
from firedrake import COMM_WORLD as comm, conditional, ConvergenceError
from numpy import isclose, squeeze, zeros
from scipy.optimize import minimize
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.io.basicio import parallel_print as pprint


def wave_dict(element_geometry, dimension, layer_shape, degree_layer, homogeneous):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    layer_shape : `str`.
        Shape of the absorbing layer, either "rectangular or "hypershape".
   degree_layer : `int` or `float` or `None`
        Hypershape degree. `None` is used only for rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    """

    dictionary = {}
    # Define options for the model. We specify the cell type, variant,
    # degree, dimension and analysis type.
    dictionary["options"] = {
        "cell_type": element_geometry,  # Options: tri/tetra(T) or quad/hexa(Q)
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        "degree": 4 if dimension == 2 else 3,  # p <= 4 for 2D and p <= 3 for 3D
        "dimension": dimension,  # Model dimension
        "analysis": "modal",  # Options: transient, modal or eikonal
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL. Here we'll assume a domain
    # with a width and depth of 1 km, and a thickness of 1 km for the 3D case.
    if dimension == 2:
        length_z, length_x, length_y = [1., 1., 0.]
    elif dimension == 3:
        length_z, length_x, length_y = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "length_z": length_z,  # depth in km - always positive
        "length_x": length_x,  # width in km - always positive
        "length_y": length_y,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_locations": ([(-length_z / 2., length_x / 4.)] if dimension == 2
                             else [(-length_z / 2., length_x / 4., length_y / 2.)]),
        "frequency": 5.,  # in Hz
        "receiver_locations": ([(-length_z, 0.),
                                (-length_z, length_x),
                                (0., 0.), (0., length_x)]
                               if dimension == 2
                               else [(-length_z, 0., 0.),
                                     (-length_z, length_x, 0.),
                                     (0., 0., 0),
                                     (0., length_x, 0.),
                                     (-length_z, 0., length_y),
                                     (-length_z, length_x, length_y),
                                     (0., 0., length_y),
                                     (0., length_x, length_y)])
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "abc_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Hom" if homogeneous else "Het")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/modal_analytical_test{dimension}d"
        + f"/modal_analytical_test{dimension}d" + str_ele
    }

    return dictionary


@fixture(scope="function")
def wave_instance(element_geometry, dimension, degree_layer, homogeneous):
    """Create an instance of the acoustic wave solver.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    degree_layer : `int` or `float` or `None`
        Hypershape degree. `None` is used only for rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

    Returns
    -------
    wave : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    fitting_c : `tuple`
        Parameters for fitting equivalent velocity regression.
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1 if dimension == 2 else 0.15

    # f_est: Factor for the stabilizing term in Eikonal equation
    # fitting_c: Parameters for fitting equivalent velocity regression
    if dimension == 2:
        if element_geometry == "T":
            f_est = 0.01 if homogeneous else 0.06
            fitting_c = (0.0, 0.0, 0.0, 0.0) if homogeneous else (0.5, 0.3, -2.2, -1.3)

    if dimension == 3:
        if element_geometry == "T":
            f_est = 0.02 if homogeneous else 0.05
            fitting_c = (0.0, 0.0, 0.0, 0.0) if homogeneous else (0.4, 0.2, 0.5, -1.0)

        else:
            f_est = 0.02 if homogeneous else 0.08
            fitting_c = (0.0, 0.0, 0.0, 0.0) if homogeneous else (0.3, 0.0, 0.5, -1.0)

    # Layer shape
    layer_shape = "rectangular" if degree_layer is None else "hypershape"

    # Get simulation parameters
    pprint(f"\nMesh Size: {1e3 * edge_length:.4f} m", comm=comm)
    pprint(f"Element Geometry: {element_geometry}", comm=comm)
    pprint(f"Eikonal Stabilizing Factor: {f_est:.2f}", comm=comm)
    pprint(f"Layer Shape: {layer_shape}", comm=comm)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    pprint((fit_str + "{:.1f}\n").format(*fitting_c), comm=comm)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(
        element_geometry, dimension, layer_shape, degree_layer, homogeneous)

    # ============ MESH FEATURES ============

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    if homogeneous:
        wave.initialize_model_parameters(constant=1.5)

    else:
        cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
        wave.initialize_model_parameters(conditional=cond)

    # Preamble mesh operations
    wave.mesh_ops.preamble_mesh_operations(wave, f_est=f_est)

    # ============ EIKONAL ANALYSIS ============

    # Finding critical points
    wave.layer_ops.critical_boundary_points(wave)

    return wave, fitting_c


def run_modal(wave, fitting_c, exp_value, n_root=1):
    """Solve the eigenvalue problem for models 2D and 3D.

    Parameters
    ----------
    wave : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
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
        n-th Root selected as the size of the absorbing layer. Default is 1.

    Returns
    -------
    None
    """

    # Determining layer size
    wave.layer_ops.layer_size_criterion(wave.mesh_parameters.lmin, n_root=n_root)

    # Creating mesh with absorbing layer
    wave.layer_ops.create_mesh_with_layer(wave)

    # Updating velocity model
    wave.layer_ops.velocity_abc(wave)

    # Modal solver
    modal_solver = 'ANALYTICAL'
    pprint(f"\nModal Solver: {modal_solver}", comm=comm)

    iter_count = [0]

    def fun_for_freq(x, par_obj):
        """Optimization sub-problem for the fundamental frequency.

        Parameters
        ----------
        x : `array`
            Design variable.
        par_obj : `list`
            Parameters for the optimization sub-problem.
            Structure: [wave, exp_value]

        Returns
        -------
        J : `float`
            Objective function of the optimization sub-problem.
        """

        iter_count[0] += 1
        print(f"Iteration: {iter_count[0]}")

        # Parameters for the optimization sub-problem
        wave, exp_value = par_obj

        # Design variables
        fitting_c = tuple(x)
        fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.3f}, "
        pprint((fit_str + "{:.3f}\n").format(*fitting_c), comm=comm)

        # Computing fundamental frequency
        wave.layer_ops.fundamental_frequency(
            wave, method="ANALYTICAL", fitting_c=fitting_c,
        )

        # Objective linearized function and its gradient
        # J = (wave.fundam_freq - exp_value)**2
        J = (wave.fundam_freq - 0.95 * exp_value)\
            * (wave.fundam_freq - 1.05 * exp_value)

        return J

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Optimization parameters
    user_tol = 1e-6
    user_maxit = 15
    method_opt = 'SLSQP'  # 'SLSQP' (13.277) # 'COBYQA' (37.156) # 'L-BFGS-B' (38.165)
    options = {'gtol': min(user_tol, 1e-6),
               'fatol': min(1e1 * user_tol, 1e-5),
               'ftol': min(1e1 * user_tol, 1e-5),
               'xatol': min(1e2 * user_tol, 1e-4),
               'xtol': min(1e2 * user_tol, 1e-4),
               'catol': min(1e3 * user_tol, 1e-3),
               'maxiter': max(user_maxit, 15),
               'maxfev': max(user_maxit, 15),
               'maxfun': max(user_maxit, 15),
               'maxls': 5,  # Maximum number of line search steps
               'norm': 2,
               'adaptive': True,
               'rhobeg': 1.0,
               'eps': 1e-12,
               'disp': False,
               }
    result = minimize(fun_for_freq, zeros(4), args=([wave, exp_value]),
                      jac='2-point', method=method_opt, options=options)

    # Optimized fitting parameters for the analytical solver
    fitting_c = tuple(squeeze(result.x))

    # Estimating computational resource usage
    name_cost = wave.path_case_abc + modal_solver + "_"
    comp_cost("tfin", tRef=tRef, user_name=name_cost)

    tol = 0.05
    fit_str = "Optimized Fitting Parameters for Analytical Solver: " + 3 * "{:.3f}, "
    pprint((fit_str + "{:.3f}\n").format(*fitting_c), comm=comm)
    abc_str = wave.case_abc if wave.layer_ops.layer_geometry.n_hyp is None \
        else f"{wave.case_abc[:2]}" + \
        f"{wave.layer_ops.layer_geometry.n_hyp:.1f}{wave.case_abc[-4:]}"
    met_str = f"Fundamental Frequency {abc_str} {wave.dimension}D. "
    met_str += f"Method {modal_solver}"
    cmp_str = f"Expected {exp_value:.5f}, got = {wave.fundam_freq:.5f}"
    assert isclose(wave.fundam_freq / exp_value, 1., atol=tol), \
        "✗ " + met_str + "  → " + cmp_str
    pprint("✓ " + met_str + " Verified: " + cmp_str, comm=comm)


@mark.older_firedrake
@mark.parametrize("element_geometry, dimension, degree_layer, homogeneous", [
    ("T", 2, 2.5, True),
    ("T", 2, None, True),
    ("T", 2, 2.0, False),
    ("T", 2, None, False),
    ("T", 3, None, True),
    ("Q", 3, None, True),
    param("T", 3, 6.0, True, marks=mark.slow),
    param("T", 3, 2.4, False, marks=mark.slow),
    param("T", 3, None, False, marks=mark.slow),
    param("Q", 3, None, False, marks=mark.slow)])
def test_modal(wave_instance, element_geometry, dimension, degree_layer, homogeneous):
    """Testing modal solvers for 2D and 3D case in Fig. 8 of Salas et al (2022).

    See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
    layers with non-reflecting boundary conditions in scalar wave equations.
    doi: https://doi.org/10.1016/j.apm.2022.09.014

    Parameters
    ----------
    wave_instance : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    degree_layer : `int` or `float` or `None`
        Hypershape degree. `None` is used only for rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

    Returns
    -------
    None

    ===================================================
    Natural Frequency for 2D model Δx = 100m - Ele = T
    ===================================================
    *EIKONAL HOMOGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.01  128.447*
     0.02  145.478

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.5 iter      (texe/pmem)     REC iter      (texe/pmem)
    ANALYTICAL    0.51121   20 (3.897s/6.152MB) 0.46875    5 (1.115s/3.061MB)
    ANALYTICAL    0.50934  usr (0.359s/2.160MB) 0.46875  usr (0.352s/2.146MB)
    KRYLOVSCH_CG  0.51046  n/a (0.038s/0.070MB) 0.46875  n/a (0.036s/0.071MB)
    RAYLEIGH      0.51084  n/a (1.328s/3.098MB) 0.46875  n/a (1.335s/2.909MB)

    *ANALYTICAL
       Case      REC*   N2.5*
    fnum[Hz]  0.46875 0.51121
    fana[Hz]  0.46875 0.51046
    fray[Hz]  0.46875 0.51084

    *EIKONAL HETEROGENEOUS
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

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.0 iter      (texe/pmem)     REC iter      (texe/pmem)
    ANALYTICAL    0.50461   35 (0.462s/3.152MB) 0.45574   40 (8.095s/8.566MB)
    ANALYTICAL    0.50428  usr (0.462s/3.152MB) 0.45737  usr (0.652s/2.990MB)
    KRYLOVSCH_CG  0.50440  n/a (0.040s/0.072MB) 0.45539  n/a (0.044s/0.072MB)
    RAYLEIGH      0.52768  n/a (1.474s/3.345MB) 0.47634  n/a (1.612s/3.455MB)

    *ANALYTICAL
       Case      REC*   N2.0*
    fnum[Hz]  0.45539 0.50440
    fana[Hz]  0.45574 0.50461
    fray[Hz]  0.47634 0.52768

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = T
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.02  146.002*
     0.03  153.839

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]    N6.0 iter        (texe/pmem)     REC iter        (texe/pmem)
    ANALYTICAL    0.52342   16 (34.635s/27.926MB) 0.47727    5 (13.720s/12.9155MB)
    ANALYTICAL    0.51628  usr ( 2.918s/ 5.911MB) 0.47727  usr ( 3.259s/ 5.725MB)
    KRYLOVSCH_CH  0.52345  n/a ( 9.954s/ 0.936MB) 0.47727  n/a (14.255s/ 0.925MB)
    RAYLEIGH      0.52678  n/a (27.041s/45.495MB) 0.47727  n/a (30.821s/52.629MB)

    *ANALYTICAL
        Case      REC*  N6.0*
    fnum[Hz]  0.47727 0.52345
    fana[Hz]  0.47727 0.52342
    fray[Hz]  0.47727 0.52678

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.03 76.777
     0.04 79.409
     0.05 82.273*
     0.06 85.347

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.4 iter         (texe/pmem)     REC iter          (texe/pmem)
    ANALYTICAL    0.51653   30 (120.432s/82.775MB) 0.42568   26 (177.947s/ 74.301MB)
    ANALYTICAL    0.51787  usr (  4.430s/10.494MB) 0.41840  usr (  6.531s/ 12.903MB)
    KRYLOVSCH_GH  0.51535  n/a ( 25.103s/ 0.077MB) 0.42562  n/a ( 64.295s/  0.075MB)
    RAYLEIGH      0.54073  n/a ( 37.131s/71.052MB) 0.44257  n/a ( 47.741s/104.142MB)

    *ANALYTICAL
       Case      REC*  N2.4*
    fnum[Hz]  0.42562 0.51535
    fana[Hz]  0.42568 0.51653
    fray[Hz]  0.44257 0.54073

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = Q
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.02  138.931*
     0.02  142.020

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter        (texe/pmem)
    ANALYTICAL    0.47741    5 ( 7.483s/14.672MB)
    ANALYTICAL    0.47741  usr ( 2.239s/ 9.275MB)
    KRYLOVSCH_GG  0.47727  n/a ( 4.860s/ 0.076MB)
    RAYLEIGH      0.47727  n/a (28.788s/30.570MB)

        Case      REC*
    fnum[Hz]  0.47741
    fana[Hz]  0.47727
    fray[Hz]  0.47727

    *EIKONAL HETEROGENEOUS
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

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter        (texe/pmem)
    ANALYTICAL    0.41350   25 (79.510s/42.848MB)
    ANALYTICAL    0.42873  usr ( 3.689s/12.027MB)
    KRYLOVSCH_GG  0.41127  n/a (25.221s/ 0.086MB)
    RAYLEIGH      0.42935  n/a (31.954s/51.852MB)

    ANALYTICAL
       Case      REC*
    fnum[Hz]  0.41127
    fana[Hz]  0.41350
    fray[Hz]  0.42935
    """

    c_hom = "Homogeneous" if homogeneous else "Heterogeneous"
    n_hyp = f"HyperShape N{degree_layer}" if degree_layer is not None else "Rectangular"

    pprint("\n" + 60 * "=" + f"\nTesting Modal Solvers with {element_geometry} elements"
           + f"for {dimension}D case\nand {n_hyp} layer. Propagation Speed: {c_hom}\n"
           + 60 * "=", comm=comm)

    # ============ SIMULATION PARAMETERS ============

    wave, fitting_c = wave_instance

    # ============ EXPECTED VALUES ============

    if dimension == 2:
        if homogeneous:
            exp_value = 0.46875 if wave.abc_deg_layer is None else 0.51046

        else:
            exp_value = 0.45539 if wave.abc_deg_layer is None else 0.50440

    if dimension == 3:
        if element_geometry == "T":
            if homogeneous:
                exp_value = 0.47727 if wave.abc_deg_layer is None else 0.52345
            else:
                exp_value = 0.42562 if wave.abc_deg_layer is None else 0.51535
        else:
            if homogeneous:
                exp_value = 0.47727
            else:
                exp_value = 0.41127

    try:
        # Computing the fundamental frequency
        run_modal(wave, fitting_c, exp_value)

        # Renaming the folder if degree_layer is modified
        wave.layer_ops.rename_folder_habc()

    except ConvergenceError as e:
        fail(f"Checking Modal Solvers with {element_geometry} elements "
             f"for{dimension}D case, {n_hyp} layer and {c_hom} propagation "
             f"speed raised an exception: {str(e)}")
