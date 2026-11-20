"""Unit tests for the Analytical Modal solver in spyro.solvers.modal.modal_ana_sol.

These tests verify the analytical modal solver by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from unittest.mock import patch
from pytest import fail, fixture, mark, skip
from firedrake import COMM_WORLD as comm, conditional, ConvergenceError
from numpy import isclose, squeeze, zeros
from scipy.optimize import minimize
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.modal.modal_ana_sol import Modal_Analytical_Solver
from spyro.utils.cost import comp_cost
from spyro.io.basicio import parallel_print as pprint
from tests.on_one_core.test_eikonal import case_parameter

pytestmark = mark.modal

# Simulation parameters and expected values. Each entry is a decision tree over
# the case fields (dimension, element_geometry, homogeneous, layer_shape): every
# nested dictionary splits on exactly one field, keyed by that field's options
# (2/3, "T"/"Q", True/False and "rectangular"/"hypershape"), and each leaf is
# the value shared by all the cases below it. A field is omitted where all its
# options share the same value, so that value is set once for the whole
# subtree. See case_parameter in test_eikonal.py for the lookup.
MODAL_CONFIG = {
    # Hypershape degree of the absorbing layer, None for rectangular layers. Only
    # T elements are tested with hypershape layers
    "degree_layer": {
        "rectangular": None,
        "hypershape": {2: {True: 2.5, False: 2.0}, 3: {True: 6.0, False: 2.4}},
    },
    # Mesh size in km: edge_length = lba / cpw, where cpw is the number of cells
    # per wavelength and lba = minimum_velocity / source_frequency
    "edge_length": {2: 0.1, 3: 0.15},
    # Factor for the stabilizing term in Eikonal equation, the value of
    # test_modal_solvers.py, which converges with every supported Firedrake
    # version (see Notes)
    "f_est": {2: {"T": 0.06, "Q": 0.05}, 3: {"T": 0.05, "Q": 0.08}},
    # Parameters for fitting equivalent velocity regression, those calibrated in
    # test_modal_solvers.py, only reported ("usr" rows of the docstring); the
    # test optimizes them from zero. The homogeneous model uses (0, 0, 0, 0), the
    # default of the analytical solver
    "fitting_c": {
        True: (0.0, 0.0, 0.0, 0.0),
        False: {2: (-0.5, 0.3, -1.5, -0.9), 3: (-0.9, 0.3, -2.1, -1.5)},
    },
    # Expected fundamental frequency in Hz, fnum of the FEM solvers (see Notes)
    "exp_value": {
        2: {
            True: {"rectangular": 0.46875, "hypershape": 0.51046},
            False: {"rectangular": {"T": 0.45539, "Q": 0.45542}, "hypershape": 0.50440},
        },
        3: {
            True: {"rectangular": 0.47727, "hypershape": 0.52345},
            False: {"rectangular": {"T": 0.42562, "Q": 0.41127}, "hypershape": 0.51535},
        },
    },
}


def wave_dict(element_geometry, dimension, layer_shape, degree_layer, homogeneous):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    layer_shape : `str`
        Shape of the absorbing layer, either "rectangular" or "hypershape".
    degree_layer : `int` or `float` or `None`
        Hypershape degree. `None` is used only for rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

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
        length_z, length_x, length_y = [1.0, 1.0, 0.0]
    elif dimension == 3:
        length_z, length_x, length_y = [1.0, 1.0, 1.0]  # in km
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
        "source_locations": (
            [(-length_z / 2.0, length_x / 4.0)]
            if dimension == 2
            else [(-length_z / 2.0, length_x / 4.0, length_y / 2.0)]
        ),
        "frequency": 5.0,  # in Hz
        "receiver_locations": (
            [(-length_z, 0.0), (-length_z, length_x), (0.0, 0.0), (0.0, length_x)]
            if dimension == 2
            else [
                (-length_z, 0.0, 0.0),
                (-length_z, length_x, 0.0),
                (0.0, 0.0, 0),
                (0.0, length_x, 0.0),
                (-length_z, 0.0, length_y),
                (-length_z, length_x, length_y),
                (0.0, 0.0, length_y),
                (0.0, length_x, length_y),
            ]
        ),
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
        + f"/modal_analytical_test{dimension}d"
        + str_ele
    }

    return dictionary


@fixture(scope="function")
def wave_instance(element_geometry, dimension, layer_shape, homogeneous):
    """Create an instance of the acoustic wave solver.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    layer_shape : `str`
        Shape of the absorbing layer. Options: "rectangular" or "hypershape".
        The hypershape degree is set in MODAL_CONFIG. Q elements are only
        tested with rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

    Returns
    -------
    wave : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    fitting_c : `tuple`
        Parameters for fitting equivalent velocity regression, those of
        test_modal_solvers.py. (0, 0, 0, 0) for the homogeneous model, the
        default of the analytical solver.
    """

    # ============ SIMULATION PARAMETERS ============

    # Only T elements are tested with hypershape layers
    if element_geometry == "Q" and layer_shape == "hypershape":
        skip("Hypershape layers are only tested with T elements")

    # Hypershape degree, mesh size (in km), factor for the stabilizing term in
    # Eikonal equation and fitting_c of the case (see MODAL_CONFIG)
    case = (dimension, element_geometry, homogeneous, layer_shape)
    degree_layer = case_parameter(MODAL_CONFIG["degree_layer"], case)
    edge_length = case_parameter(MODAL_CONFIG["edge_length"], case)
    f_est = case_parameter(MODAL_CONFIG["f_est"], case)
    fitting_c = case_parameter(MODAL_CONFIG["fitting_c"], case)

    # Get simulation parameters
    pprint(f"\nMesh Size: {1e3 * edge_length:.4f} m", comm=comm)
    pprint(f"Element Geometry: {element_geometry}", comm=comm)
    pprint(f"Eikonal Stabilizing Factor: {f_est:.2f}", comm=comm)
    pprint(f"Layer Shape: {layer_shape}", comm=comm)
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    pprint((fit_str + "{:.1f}\n").format(*fitting_c), comm=comm)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(
        element_geometry, dimension, layer_shape, degree_layer, homogeneous
    )

    # ============ MESH FEATURES ============

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    if homogeneous:
        wave.set_initial_velocity_model(constant=1.5)

    else:
        cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
        wave.set_initial_velocity_model(conditional=cond)

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
    fitting_c : `tuple`
        Parameters for fitting equivalent velocity regression, only reported.
        (0, 0, 0, 0) for the homogeneous model.
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

    Notes
    -----
    The optimization always starts from fitting_c = (0, 0, 0, 0), so the
    test checks that a fit is found without a prior guess. The 'fitting_c'
    argument is only reported, and is not the initial guess.

    The optimizer evaluates the ANALYTICAL solver once per iteration, for a
    new fitting_c. The equivalent velocities c_eq and c_eqref do not depend
    on fitting_c, only on the model, and computing them takes two static
    solves. They are therefore computed at the first evaluation and reused
    for the rest of the optimization of this model; every evaluation still
    runs the regression and the frequency of the ANALYTICAL solver.

    Choice of SLSQP. The scipy methods were compared on the 12 cases with the
    options below (at most 15 iterations or evaluations), serially on one
    core with Firedrake 2025.4.1: relative error abs(fana - fexp) / fexp at
    the returned fitting_c, evaluations of the ANALYTICAL solver and time of
    the optimization, both summed over the cases, and failed cases.

    Method       Max error  Mean error  Evaluations    Time   Failed
    SLSQP           1.37 %      0.16 %          250  10.8 s        0
    L-BFGS-B        0.81 %      0.23 %          205   9.6 s        0
    COBYQA          0.90 %      0.30 %          180   9.7 s        0
    COBYLA          1.54 %      0.44 %          180   9.1 s        0
    TNC             5.57 %      0.70 %          595  14.1 s        1
    Nelder-Mead    62.85 %     18.99 %          180   9.3 s        6
    Powell         47.19 %     14.76 %          180   9.4 s       10

    SLSQP is the most accurate: every heterogeneous case ends within
    0.18 %, and its largest error is T-3D N6.0 homogeneous, which stops at
    the start (see the Notes of test_modal). L-BFGS-B, COBYQA and COBYLA stop
    at their evaluation limit in most cases, with heterogeneous errors up to
    0.8 % to 1.5 %. TNC needs the most evaluations and fails Q-3D
    heterogeneous, and Nelder-Mead and Powell barely move from (0, 0, 0, 0)
    in 15 evaluations. The test checks the frequency of the last evaluation,
    which for these methods can differ from the returned fitting_c, so they
    fail more cases than their errors alone suggest. The cost differs by
    less than 2 s, in a file that takes about 52 s, most of it in building
    the meshes.
    """

    # Determining layer size
    wave.layer_ops.layer_size_criterion(wave.mesh_parameters.lmin, n_root=n_root)

    # Creating mesh with absorbing layer
    wave.layer_ops.create_mesh_with_layer(wave)

    # Updating velocity model
    wave.layer_ops.velocity_abc(wave)

    # Modal solver
    modal_solver = "ANALYTICAL"
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
            wave,
            method="ANALYTICAL",
            fitting_c=fitting_c,
        )

        # Objective linearized function and its gradient
        # J = (wave.fundam_freq - exp_value)**2
        J = (wave.fundam_freq - 0.95 * exp_value) * (
            wave.fundam_freq - 1.05 * exp_value
        )

        return J

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Optimization parameters. The fit always starts from zero, not from the
    # fitting_c of the fixture
    fitting_c_ini = zeros(4)
    user_tol = 1e-6
    user_maxit = 15
    method_opt = "SLSQP"  # The most accurate scipy method here (see Notes)
    options = {
        "gtol": min(user_tol, 1e-6),
        "fatol": min(1e1 * user_tol, 1e-5),
        "ftol": min(1e1 * user_tol, 1e-5),
        "xatol": min(1e2 * user_tol, 1e-4),
        "xtol": min(1e2 * user_tol, 1e-4),
        "catol": min(1e3 * user_tol, 1e-3),
        "maxiter": max(user_maxit, 15),
        "maxfev": max(user_maxit, 15),
        "maxfun": max(user_maxit, 15),
        "maxls": 5,  # Maximum number of line search steps
        "norm": 2,
        "adaptive": True,
        "rhobeg": 1.0,
        "eps": 1e-12,
        "disp": False,
    }
    # Equivalent velocities of this model, computed at the first evaluation
    homogenized = []
    homogenized_velocities = Modal_Analytical_Solver.homogenized_velocities

    def homogenized_velocities_once(
        self: Modal_Analytical_Solver, *args, **kwargs
    ) -> tuple:
        """Return the equivalent velocities, computing them only once.

        Parameters
        ----------
        self : `modal_ana_sol.Modal_Analytical_Solver`
            Analytical solver of the current evaluation.
        *args
            Positional arguments of `homogenized_velocities`.
        **kwargs
            Keyword arguments of `homogenized_velocities`.

        Returns
        -------
        `tuple`
            Equivalent velocities ``(c_eq, c_eqref)`` of the model.
        """
        if not homogenized:
            homogenized.append(homogenized_velocities(self, *args, **kwargs))
        return homogenized[0]

    with patch.object(
        Modal_Analytical_Solver, "homogenized_velocities", homogenized_velocities_once
    ):
        result = minimize(
            fun_for_freq,
            fitting_c_ini,
            args=([wave, exp_value]),
            jac="2-point",
            method=method_opt,
            options=options,
        )

    # Optimized fitting parameters for the analytical solver
    fitting_c = tuple(squeeze(result.x))

    # Estimating computational resource usage
    name_cost = wave.path_case_abc + modal_solver + "_"
    comp_cost("tfin", tRef=tRef, user_name=name_cost)

    tol = 0.05
    fit_str = "Optimized Fitting Parameters for Analytical Solver: " + 3 * "{:.3f}, "
    pprint((fit_str + "{:.3f}\n").format(*fitting_c), comm=comm)
    abc_str = (
        wave.case_abc
        if wave.layer_ops.layer_geometry.n_hyp is None
        else f"{wave.case_abc[:2]}"
        + f"{wave.layer_ops.layer_geometry.n_hyp:.1f}{wave.case_abc[-4:]}"
    )
    met_str = f"Fundamental Frequency {abc_str} {wave.dimension}D. "
    met_str += f"Method {modal_solver}"
    cmp_str = f"Expected {exp_value:.5f}, got = {wave.fundam_freq:.5f}"
    assert isclose(wave.fundam_freq / exp_value, 1.0, atol=tol), (
        "✗ " + met_str + "  → " + cmp_str
    )
    pprint("✓ " + met_str + " Verified: " + cmp_str, comm=comm)


@mark.parametrize("homogeneous", [True, False])
@mark.parametrize("layer_shape", ["hypershape", "rectangular"])
@mark.parametrize("dimension", [2, 3])
@mark.parametrize("element_geometry", ["T", "Q"])
def test_modal(wave_instance, element_geometry, dimension, layer_shape, homogeneous):
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
    layer_shape : `str`
        Shape of the absorbing layer. Options: "rectangular" or "hypershape".
        The hypershape degree is set in MODAL_CONFIG. Q elements are only
        tested with rectangular layers.
    homogeneous : `bool`
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

    Returns
    -------
    None

    Notes
    -----
    Theoretical eik_min: 166.667 ms in the homogeneous model and 83.333 ms in
    the heterogeneous one, on x = 0 (see test_modal_solvers.py). Every listed
    f_est converges without retries with Firedrake 2025.4.1. The marked f_est
    (*) is the one used by the fixture: the value with the smallest relative
    error abs(eik_min - eik_thr) / eik_thr among those that converge with every
    supported Firedrake version (2025.4.1 and 2026.10.0), as in
    test_eikonal.py. It equals the f_est of test_modal_solvers.py, whose meshes
    are the same. #: converges with 2025.4.1 but not with 2026.10.0, where the
    solver retries with the next f_est.

    The homogeneous f_est were 0.01 (2D) and 0.02 (3D), far from eik_thr, and
    0.02 does not converge for Q-3D with 2026.10.0. They were replaced by the
    values selected above. The final layered meshes are the same, so every
    frequency is unchanged.

    In the RESULTS tables, the first ANALYTICAL row is the result of the
    test: fitting_c is optimized with SLSQP from (0, 0, 0, 0), iter is the
    number of evaluations of the objective, and texe/pmem cover the whole
    optimization. The equivalent velocities are computed once per model (see
    run_modal). The second ANALYTICAL row (usr) uses the fitting_c of the
    fixture, without optimization ((0, 0, 0, 0) for the homogeneous
    models). The KRYLOVSCH and RAYLEIGH rows are references, which the test
    does not run (n/a: no optimization). fnum, fana and fray are the FEM,
    optimized ANALYTICAL and RAYLEIGH frequencies. texe and pmem are the
    runtime and the peak memory traced by Python, measured serially on one
    core with Firedrake 2025.4.1 and the compilation caches already filled.


    The fitting_c of the fixture, used without optimization in the usr rows,
    are those calibrated in test_modal_solvers.py, (-0.5, 0.3, -1.5, -0.9)
    in 2D and (-0.9, 0.3, -2.1, -1.5) in 3D. The heterogeneous models of
    both tests are the same, and the heterogeneous usr rows are within
    0.73 % of fnum. The homogeneous models use (0, 0, 0, 0), the default of
    the analytical solver. The optimized fitting_c, found for each model
    separately, are within 0.18 % of fnum in the heterogeneous models. They
    differ from the calibrated ones, since four parameters are fitted to a
    single frequency and many values do so. The objective
    (f - 0.95 fexp) (f - 1.05 fexp) is smallest at f = fexp, and SLSQP stops
    at the first values that reach it within its tolerance.

    Of the homogeneous models, only T-2D N2.5 moves away from (0, 0, 0, 0),
    to (0.010, 0.008, -0.022, -0.113). The others stop after the first
    gradient (iter = 5: one evaluation and four finite differences): with an
    identity initial Hessian the first step of SLSQP is -g, and SLSQP takes
    the start as converged when the predicted change |g|^2 is below
    ftol = 1e-5. In the rectangular layers, (0, 0, 0, 0) already gives fexp
    within 0.03 %, so the gradient is nearly zero (|g|^2 <= 1.6e-9).
    T-3D N6.0 stays at -1.37 %, although fp1 and fp2 change its frequency:
    the objective is of order 1e-4, so its gradient (0, 0, -1.6e-3, -7.5e-4)
    gives |g|^2 = 3.1e-6, below ftol.

    ===================================================
    Natural Frequency for 2D model Δx = 100m - Ele = T
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.01  128.447
     0.02  145.478
     0.03  154.439
     0.04  159.731
     0.05  163.373
     0.06  166.193*
     0.07  168.555

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.5 iter      (texe/pmem)     REC iter      (texe/pmem)
    ANALYTICAL    0.51057   20 (0.474s/1.464MB) 0.46875    5 (0.246s/0.613MB)
    ANALYTICAL    0.52430  usr (0.142s/0.389MB) 0.46875  usr (0.136s/0.381MB)
    KRYLOVSCH_CG  0.51046  n/a (0.058s/0.124MB) 0.46875  n/a (0.064s/0.157MB)
    RAYLEIGH      0.51084  n/a (0.180s/0.787MB) 0.46875  n/a (0.178s/0.817MB)

    *ANALYTICAL
       Case     N2.5*     REC*
    fnum[Hz]  0.51046  0.46875
    fana[Hz]  0.51057  0.46875
    fray[Hz]  0.51084  0.46875

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    N2.5  optimized   0.010   0.008  -0.022  -0.113
    N2.5  usr         0.000   0.000   0.000   0.000
    REC   optimized   0.000   0.000   0.000   0.000
    REC   usr         0.000   0.000   0.000   0.000

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.01   66.836
     0.02   73.308
     0.03   77.178
     0.04   79.680
     0.05   81.498
     0.06   82.942*
     0.07   84.160
     0.08   85.233

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.0 iter      (texe/pmem)     REC iter      (texe/pmem)
    ANALYTICAL    0.50473   30 (1.096s/2.810MB) 0.45572   35 (0.881s/2.416MB)
    ANALYTICAL    0.50526  usr (0.216s/0.477MB) 0.45256  usr (0.217s/0.449MB)
    KRYLOVSCH_CG  0.50440  n/a (0.079s/0.138MB) 0.45539  n/a (0.078s/0.113MB)
    RAYLEIGH      0.52768  n/a (0.169s/1.103MB) 0.47634  n/a (0.188s/1.130MB)

    *ANALYTICAL
       Case     N2.0*     REC*
    fnum[Hz]  0.50440  0.45539
    fana[Hz]  0.50473  0.45572
    fray[Hz]  0.52768  0.47634

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    N2.0  optimized   0.277   0.006   0.000  -0.551
    N2.0  usr        -0.500   0.300  -1.500  -0.900
    REC   optimized   0.569   0.174  -0.640   0.000
    REC   usr        -0.500   0.300  -1.500  -0.900

    ===================================================
    Natural Frequency for 2D model Δx = 100m - Ele = Q
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.01  129.509
     0.02  151.709
     0.03  159.931
     0.04  164.457
     0.05  167.703*
     0.06  170.399
     0.07  172.805

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter      (texe/pmem)
    ANALYTICAL    0.46875    5 (0.240s/0.600MB)
    ANALYTICAL    0.46875  usr (0.126s/0.475MB)
    KRYLOVSCH_CG  0.46875  n/a (0.063s/0.125MB)
    RAYLEIGH      0.46875  n/a (0.163s/0.713MB)

    *ANALYTICAL
       Case      REC*
    fnum[Hz]  0.46875
    fana[Hz]  0.46875
    fray[Hz]  0.46875

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    REC   optimized   0.000   0.000   0.000   0.000
    REC   usr         0.000   0.000   0.000   0.000

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.01   65.002
     0.02   75.811
     0.03   79.845
     0.04   82.101
     0.05   83.744*
     0.06   85.118
     0.07   86.345
     0.08   87.480

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter      (texe/pmem)
    ANALYTICAL    0.45625   35 (0.979s/3.144MB)
    ANALYTICAL    0.45832  usr (0.206s/0.480MB)
    KRYLOVSCH_CG  0.45542  n/a (0.077s/0.141MB)
    RAYLEIGH      0.47619  n/a (0.196s/0.913MB)

    *ANALYTICAL
       Case      REC*
    fnum[Hz]  0.45542
    fana[Hz]  0.45625
    fray[Hz]  0.47619

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    REC   optimized   0.056   0.974  -0.887   0.000
    REC   usr        -0.500   0.300  -1.500  -0.900

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = T
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.02  146.002
     0.03  153.839
     0.04  159.016
     0.05  164.698*
     0.06  170.789

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]    N6.0 iter       (texe/pmem)     REC iter       (texe/pmem)
    ANALYTICAL    0.51628    5 (0.898s/ 4.667MB) 0.47741    5 (0.764s/ 2.488MB)
    ANALYTICAL    0.51628  usr (0.557s/ 2.272MB) 0.47741  usr (0.513s/ 2.381MB)
    KRYLOVSCH_CH  0.52345  n/a (3.201s/ 0.123MB) 0.47727  n/a (4.896s/ 0.123MB)
    RAYLEIGH      0.52678  n/a (2.000s/35.738MB) 0.47727  n/a (2.432s/42.724MB)

    *ANALYTICAL
       Case     N6.0*     REC*
    fnum[Hz]  0.52345  0.47727
    fana[Hz]  0.51628  0.47741
    fray[Hz]  0.52678  0.47727

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    N6.0  optimized   0.000   0.000   0.000   0.000
    N6.0  usr         0.000   0.000   0.000   0.000
    REC   optimized   0.000   0.000   0.000   0.000
    REC   usr         0.000   0.000   0.000   0.000

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.03   76.698
     0.04   79.370
     0.05   82.259*
     0.06   85.344

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]    N2.4 iter        (texe/pmem)     REC iter        (texe/pmem)
    ANALYTICAL    0.51555   35 ( 1.773s/ 4.043MB) 0.42584   35 ( 1.816s/ 4.424MB)
    ANALYTICAL    0.51669  usr ( 0.695s/ 2.696MB) 0.42541  usr ( 0.809s/ 3.813MB)
    KRYLOVSCH_GH  0.51535  n/a ( 6.921s/ 0.110MB) 0.42562  n/a (11.980s/ 0.136MB)
    RAYLEIGH      0.54073  n/a ( 2.819s/60.581MB) 0.44257  n/a ( 4.590s/95.062MB)

    *ANALYTICAL
       Case     N2.4*     REC*
    fnum[Hz]  0.51535  0.42562
    fana[Hz]  0.51555  0.42584
    fray[Hz]  0.54073  0.44257

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    N2.4  optimized   0.156   0.655  -0.168  -0.988
    N2.4  usr        -0.900   0.300  -2.100  -1.500
    REC   optimized   0.169   0.860  -1.032  -0.001
    REC   usr        -0.900   0.300  -2.100  -1.500

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = Q
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.02  138.931#
     0.03  142.020
     0.04  146.427
     0.05  151.578
     0.06  157.112
     0.07  162.852
     0.08  168.726*
     0.09  174.708

    *RESULTS HOMOGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter       (texe/pmem)
    ANALYTICAL    0.47741    5 (0.703s/ 3.272MB)
    ANALYTICAL    0.47741  usr (0.556s/ 3.255MB)
    KRYLOVSCH_GG  0.47727  n/a (1.761s/ 0.124MB)
    RAYLEIGH      0.47727  n/a (1.706s/16.701MB)

    *ANALYTICAL
       Case      REC*
    fnum[Hz]  0.47727
    fana[Hz]  0.47741
    fray[Hz]  0.47727

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    REC   optimized   0.000   0.000   0.000   0.000
    REC   usr         0.000   0.000   0.000   0.000

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.02   69.442
     0.03   70.974
     0.04   73.179
     0.05   75.766
     0.06   78.548
     0.07   81.431
     0.08   84.377*
     0.09   87.376

    *RESULTS HETEROGENEOUS (usr: without optimization)
    Frequency[Hz]     REC iter       (texe/pmem)
    ANALYTICAL    0.41160   35 (1.788s/ 8.184MB)
    ANALYTICAL    0.41427  usr (0.698s/ 3.499MB)
    KRYLOVSCH_GG  0.41127  n/a (5.538s/ 0.142MB)
    RAYLEIGH      0.42935  n/a (2.960s/36.020MB)

    *ANALYTICAL
       Case      REC*
    fnum[Hz]  0.41127
    fana[Hz]  0.41160
    fray[Hz]  0.42935

    *FITTING_C (optimized: SLSQP from (0, 0, 0, 0); usr: fixture)
    Case  Approach      fc1     fc2     fp1     fp2
    REC   optimized   0.184   0.857  -1.044  -0.001
    REC   usr        -0.900   0.300  -2.100  -1.500

    The tables were computed with Firedrake 2025.4.1. With Firedrake
    2026.10.0 all 12 cases pass, and every marked f_est converges without
    retries; two later runs with the final test code (12 passed, 4 skipped,
    in 55 s and 43 s) gave the same values. eik_min agrees within 0.02 ms (in ms,
    2026.10.0 vs 2025.4.1):

    Case     Homogeneous         Heterogeneous
    T-2D     166.187  166.193    82.949  82.942
    Q-2D     167.703  167.703    83.744  83.744
    T-3D     164.686  164.698    82.243  82.259
    Q-3D     168.744  168.726    84.375  84.377

    An earlier run with the previous homogeneous f_est showed that 0.02 does
    not converge for Q-3D with 2026.10.0 (marked #).

    Optimized ANALYTICAL with 2026.10.0 vs 2025.4.1 (fitting_c; fana in Hz):

    Case       Version    fitting_c                          fana
    T-2D N2.5  2026.10.0  ( 0.010,  0.003, -0.022, -0.112)  0.51112
               2025.4.1   ( 0.010,  0.008, -0.022, -0.113)  0.51057
    T-2D N2.0  2026.10.0  ( 0.277,  0.006,  0.000, -0.551)  0.50473
               2025.4.1   ( 0.277,  0.006,  0.000, -0.551)  0.50473
    T-2D REC   2026.10.0  ( 0.569,  0.174, -0.640,  0.000)  0.45572
               2025.4.1   ( 0.569,  0.174, -0.640,  0.000)  0.45572
    Q-2D REC   2026.10.0  ( 0.056,  0.974, -0.887,  0.000)  0.45625
               2025.4.1   ( 0.056,  0.974, -0.887,  0.000)  0.45625
    T-3D N2.4  2026.10.0  ( 0.177,  0.696, -0.182, -1.061)  0.51566
               2025.4.1   ( 0.156,  0.655, -0.168, -0.988)  0.51555
    T-3D REC   2026.10.0  ( 0.186,  0.930, -1.119, -0.001)  0.42595
               2025.4.1   ( 0.169,  0.860, -1.032, -0.001)  0.42584
    Q-3D REC   2026.10.0  ( 0.195,  0.909, -1.107, -0.001)  0.41169
               2025.4.1   ( 0.184,  0.857, -1.044, -0.001)  0.41160

    The other homogeneous models keep (0, 0, 0, 0) and the same frequency with
    both versions. The 2D heterogeneous results are identical. In 3D the
    velocity of the cells whose centroid lies on x = 0.5 depends on round-off
    (see test_modal_solvers.py), so c_eq and the optimized fitting_c differ,
    and both fits end within 0.10 % of the expected values. In T-2D N2.5,
    c_eq = c_eqref at (0, 0, 0, 0), so both terms of the max in c_ref are
    equal and the finite-difference gradient depends on round-off: SLSQP takes
    a slightly different path, and both results are within 0.13 % of fnum.
    """

    case = (dimension, element_geometry, homogeneous, layer_shape)
    degree_layer = case_parameter(MODAL_CONFIG["degree_layer"], case)
    c_hom = "Homogeneous" if homogeneous else "Heterogeneous"
    n_hyp = f"HyperShape N{degree_layer}" if degree_layer is not None else "Rectangular"

    pprint(
        "\n"
        + 60 * "="
        + f"\nTesting Modal Solvers with {element_geometry} elements"
        + f"for {dimension}D case\nand {n_hyp} layer. Propagation Speed: {c_hom}\n"
        + 60 * "=",
        comm=comm,
    )

    # ============ SIMULATION PARAMETERS ============

    wave, fitting_c = wave_instance

    # ============ EXPECTED VALUES ============

    exp_value = case_parameter(MODAL_CONFIG["exp_value"], case)

    try:
        # Computing the fundamental frequency
        run_modal(wave, fitting_c, exp_value)

        # Renaming the folder if degree_layer is modified
        wave.layer_ops.rename_folder_habc()

    except ConvergenceError as e:
        fail(
            f"Checking Modal Solvers with {element_geometry} elements "
            f"for{dimension}D case, {n_hyp} layer and {c_hom} propagation "
            f"speed raised an exception: {str(e)}"
        )
