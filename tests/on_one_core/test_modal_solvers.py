"""Unit tests for the Modal solvers implemented in spyro.solvers.modal.modal_sol.

These tests verify the implemented modal solvers by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from pytest import fail, fixture, mark, param, skip
from firedrake import conditional, ConvergenceError
from numpy import cos, isclose, linspace, nonzero, pi, sign, sin
from scipy.optimize import brentq
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
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
    "edge_length": {2: 1.0 / 10, 3: 1.0 / 7},
    # Factor for the stabilizing term in Eikonal equation (see Notes)
    "f_est": {2: {"T": 0.06, "Q": 0.05}, 3: {"T": 0.05, "Q": 0.08}},
    # Parameters for fitting equivalent velocity regression, calibrated for the
    # static problem of c_equivalent (see Notes). The homogeneous model uses
    # (0, 0, 0, 0), the default of the analytical solver
    "fitting_c": {
        True: (0.0, 0.0, 0.0, 0.0),
        False: {2: (-0.5, 0.3, -1.5, -0.9), 3: (-0.9, 0.3, -2.1, -1.5)},
    },
    # Methods used to solve the eigenvalue problem, the same for every case
    "modal_solver_lst": [
        "ANALYTICAL",
        "ARNOLDI",
        "LANCZOS",
        "LOBPCG",
        "KRYLOVSCH_CH",
        "KRYLOVSCH_CG",
        "KRYLOVSCH_GH",
        "KRYLOVSCH_GG",
        "RAYLEIGH",
    ],
    # Expected fundamental frequency in Hz of the continuous model, independent of
    # the discretization: exact for rectangular layers (rectangular_layer_frequency,
    # checked by test_rectangular_expected_frequency) and converged on refined
    # meshes for hypershape layers (see Notes)
    "exp_value": {
        2: {
            True: {"rectangular": 0.46875, "hypershape": 0.50853},
            False: {"rectangular": 0.45613, "hypershape": 0.50193},
        },
        3: {
            True: {"rectangular": 0.47727, "hypershape": 0.50139},
            False: {"rectangular": 0.42572, "hypershape": 0.50831},
        },
    },
}


def rectangular_layer_frequency(pad: float, c_left: float, c_right: float) -> float:
    """Return the exact fundamental frequency of a model with a rectangular layer.

    Parameters
    ----------
    pad : `float`
        Size of the absorbing layer in km.
    c_left : `float`
        Velocity in km/s for x < 0.5 km, also in the layer.
    c_right : `float`
        Velocity in km/s for x >= 0.5 km, also in the layer.

    Returns
    -------
    `float`
        Fundamental frequency in Hz.

    Notes
    -----
    The modal problem is -div(c^2 grad p) = omega^2 p with Neumann conditions on
    the whole boundary, and the domain with the layer spans x in [-pad, 1 + pad].
    The velocity depends only on x, so the problem separates [2]_: each mode is
    X(x) times a cosine in z (and y) of wavenumber k, with
    -(c^2 X')' + k^2 c^2 X = omega^2 X. The fundamental is the first nonzero
    mode with k = 0, which varies only along x. In the cases of the test the
    modes with k > 0 have higher frequencies, except in the 3D homogeneous
    model, where y spans the same length as x and the first mode along y has
    the same frequency.

    With a = 0.5 + pad, the mode with k = 0 is
    p = cos(omega (x + pad) / c_left) for x < 0.5 and
    p = beta cos(omega (1 + pad - x) / c_right) for x >= 0.5, which satisfy
    the Neumann conditions at x = -pad and x = 1 + pad. At the interface
    x = 0.5 the coefficient c^2 jumps, and the transmission conditions of
    -(c^2 X')' = omega^2 X are the continuity of X and of c^2 X' [1]_.
    Eliminating beta from them gives

        c_left sin(omega a / c_left) cos(omega a / c_right)
        + c_right sin(omega a / c_right) cos(omega a / c_left) = 0,

    whose smallest positive root is the fundamental. With c_left = c_right = c
    it reduces to sin(2 omega a / c) = 0, that is f = c / (2 (1 + 2 pad)), the
    fundamental of a rigid-walled cavity of length 1 + 2 pad [3]_.

    The root lies between pi c_min / (2 a) and pi c_max / (2 a), with
    c_min = min(c_left, c_right) and c_max = max(c_left, c_right), and the
    search of the code ends 1 % above the upper bound. The eigenvalues
    omega^2 of -(c^2 X')' = omega^2 X are the max-min values of the quotient
    int c^2 X'^2 dx / int X^2 dx [2]_. The Neumann conditions are not imposed
    on the admissible functions, so these functions are the same for any c.
    Since c_min^2 <= c^2 <= c_max^2, the quotient of every function lies
    between its quotients in the homogeneous models with c_min and c_max, and
    then each eigenvalue lies between the eigenvalues with the same index in
    these models [2]_. Their first nonzero eigenvalues are
    (pi c_min / (2 a))^2 and (pi c_max / (2 a))^2, those of the cavity above.

    References
    ----------
    .. [1] C.-I. Gheorghiu and B. Zinsou, "Analytic vs. numerical solutions to
       a Sturm-Liouville transmission eigenproblem", J. Numer. Anal. Approx.
       Theory, vol. 48, no. 2, pp. 159-174, 2019. Problem
       -(alpha u')' = lambda u with alpha piecewise constant and the
       transmission conditions u(1/2-) = u(1/2+) and
       u'(1/2-) = c^2 u'(1/2+), Section 2 (with Dirichlet ends).
    .. [2] R. Courant and D. Hilbert, Methods of Mathematical Physics, vol. I,
       Interscience, New York, 1953. Separation of variables, Ch. V
       ("Vibration and Eigenvalue Problems"); max-min property of the
       eigenvalues, Ch. VI, Section 1.4, pp. 405-407 (theorem on p. 406);
       comparison of minimum problems with the same admissible functions,
       Ch. VI, Section 2.1 ("General Theorems"), p. 407.
    .. [3] L. E. Kinsler, A. R. Frey, A. B. Coppens and J. V. Sanders,
       Fundamentals of Acoustics, 4th ed., John Wiley & Sons, New York, 2000.
       Normal modes of a rigid-walled rectangular cavity.
    """
    half = 0.5 + pad

    def residual(omega: float) -> float:
        """Return the left side of the frequency equation.

        Parameters
        ----------
        omega : `float`
            Angular frequency in rad/s.

        Returns
        -------
        `float`
            Left side of the equation of the Notes.
        """
        return c_left * sin(omega * half / c_left) * cos(omega * half / c_right) + (
            c_right * sin(omega * half / c_right) * cos(omega * half / c_left)
        )

    # The residual grows from zero, so the first sign change brackets the root
    omega_max = 1.01 * pi * max(c_left, c_right) / (2.0 * half)
    grid = linspace(1e-6 * omega_max, omega_max, 10001)
    values = residual(grid)
    i = nonzero(sign(values[:-1]) != sign(values[1:]))[0][0]
    return brentq(residual, grid[i], grid[i + 1], xtol=1e-14) / (2.0 * pi)


def wave_dict(
    element_geometry: str,
    dimension: int,
    layer_shape: str,
    degree_layer: int | float | None,
    homogeneous: bool,
    degree_eikonal: int = 2,
) -> dict:
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
        If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.
    degree_eikonal : `int`, optional
        Finite element order for the Eikonal equation. Default is 2.

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
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Hom" if homogeneous else "Het")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/modal_test{dimension}d/modal_test{dimension}d"
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
        Parameters for fitting equivalent velocity regression. (0, 0, 0, 0)
        for the homogeneous model, the default of the analytical solver.
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
    """

    # ============ SIMULATION PARAMETERS ============

    # Eikonal parameters of test_eikonal.py for the source condition
    # 'nearest_node' and consistent elements (see Notes)
    source_bc = "nearest_node"
    element_type = "consistent"
    p_eik = 2

    # Only T elements are tested with hypershape layers
    if element_geometry == "Q" and layer_shape == "hypershape":
        skip("Hypershape layers are only tested with T elements")

    # Hypershape degree, mesh size (in km), factor for the stabilizing term in
    # Eikonal equation, fitting_c and modal solvers of the case (see MODAL_CONFIG)
    case = (dimension, element_geometry, homogeneous, layer_shape)
    degree_layer = case_parameter(MODAL_CONFIG["degree_layer"], case)
    edge_length = case_parameter(MODAL_CONFIG["edge_length"], case)
    f_est = case_parameter(MODAL_CONFIG["f_est"], case)
    fitting_c = case_parameter(MODAL_CONFIG["fitting_c"], case)
    modal_solver_lst = case_parameter(MODAL_CONFIG["modal_solver_lst"], case)

    # Get simulation parameters
    SpyroEnsemble.print(f"\nMesh Size: {1e3 * edge_length:.4f} m")
    SpyroEnsemble.print(f"Element Geometry: {element_geometry}")
    SpyroEnsemble.print(f"Element Type: {element_type}")
    SpyroEnsemble.print(f"Source Condition: {source_bc}")
    SpyroEnsemble.print(f"Eikonal Degree: {p_eik}")
    SpyroEnsemble.print(f"Eikonal Stabilizing Factor: {f_est:.2f}")
    SpyroEnsemble.print(f"Layer Shape: {layer_shape}")
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    SpyroEnsemble.print((fit_str + "{:.1f}\n").format(*fitting_c))

    # Create dictionary with parameters for the model
    dictionary = wave_dict(
        element_geometry,
        dimension,
        layer_shape,
        degree_layer,
        homogeneous,
        degree_eikonal=p_eik,
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
    wave.mesh_ops.preamble_mesh_operations(
        wave, ele_type_eik=element_type, f_est=f_est, source_bc_eik=source_bc
    )

    # ============ EIKONAL ANALYSIS ============

    # Finding critical points
    wave.layer_ops.critical_boundary_points(wave)

    return wave, fitting_c, modal_solver_lst


def run_modal(wave, modal_solver_lst, fitting_c, exp_value, n_root=1):
    """Solve the eigenvalue problem for models 2D and 3D.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
    fitting_c : `tuple`
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

    # Loop for different modal solvers
    for modal_solver in modal_solver_lst:
        # Modal solver
        SpyroEnsemble.print(f"\nModal Solver: {modal_solver}")

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing fundamental frequency
        wave.layer_ops.fundamental_frequency(
            wave, method=modal_solver, fitting_c=fitting_c
        )

        # Estimating computational resource usage
        name_cost = wave.path_case_abc + modal_solver + "_"
        comp_cost("tfin", tRef=tRef, user_name=name_cost)

        # Relative tolerance against the frequency of the continuous model. The
        # FEM solvers share the discretization error of the test mesh, at most
        # 4.40 % in T-3D N6.0 (see Notes)
        tol = 0.09 if modal_solver in ("ANALYTICAL", "RAYLEIGH") else 0.05

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
        SpyroEnsemble.print("✓ " + met_str + " Verified: " + cmp_str)


@mark.parametrize(
    "dimension, homogeneous, pad",
    [(2, True, 0.3), (2, False, 0.5), (3, True, 2.0 / 7.0), (3, False, 4.0 / 7.0)],
)
def test_rectangular_expected_frequency(
    dimension: int, homogeneous: bool, pad: float
) -> None:
    """Check the expected frequencies of the rectangular layers.

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem, 2 or 3.
    homogeneous : `bool`
        If `True`, the velocity is 1.5 km/s; otherwise 3.0 km/s for x < 0.5
        and 1.5 km/s for x >= 0.5.
    pad : `float`
        Size of the layer in km selected for the case by test_modal.

    Returns
    -------
    None

    Notes
    -----
    The expected value is the exact frequency of the continuous model
    (rectangular_layer_frequency), rounded to 5 decimals, and is the same for
    T and Q elements.
    """
    c_left = 1.5 if homogeneous else 3.0
    exact = round(rectangular_layer_frequency(pad, c_left, 1.5), 5)
    for element_geometry in ("T", "Q"):
        case = (dimension, element_geometry, homogeneous, "rectangular")
        assert case_parameter(MODAL_CONFIG["exp_value"], case) == exact


@mark.parametrize("homogeneous", [True, False])
@mark.parametrize("layer_shape", ["hypershape", "rectangular"])
@mark.parametrize("dimension", [2, param(3, marks=mark.slow)])
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
    Theoretical value. eik_min is the first-arrival time on the absorbing
    boundaries: z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1 (z = 0 is
    the free surface). The source is at (z, x) = (-0.5, 0.25) km in 2D and
    (z, x, y) = (-0.5, 0.25, 0.5) km in 3D. The velocity is 1.5 km/s in the
    homogeneous model. In the heterogeneous one it is 3.0 km/s for x < 0.5
    and 1.5 km/s for x >= 0.5. The first arrival on a boundary follows the
    straight ray to its nearest point:

    Boundary   Nearest point (z, x)   Homogeneous [ms]   Heterogeneous [ms]
    x = 0      (-0.5, 0.0)                     166.667               83.333
    z = -1     (-1.0, 0.25)                    333.333              166.667
    y = 0, 1   3D: (-0.5, 0.25, 0|1)           333.333              166.667
    x = 1      (-0.5, 1.0)                     500.000              416.667

    eik_min is on x = 0 at (-0.5, 0.0) km (at (-0.5, 0.0, 0.5) km in 3D):
    0.25 km / 1.5 km/s = 166.667 ms in the homogeneous model and
    0.25 km / 3.0 km/s = 83.333 ms in the heterogeneous one.

    Selection of the parameters. The Eikonal is solved as in test_eikonal.py
    for the source condition 'nearest_node' and consistent elements with
    p = 2. The meshes, edge_length = 1/10 km in 2D and 1/7 km in 3D, have
    the same 10 and 7 cells per side as the 100 m and 150 m meshes of
    test_eikonal.py. N = 1 / edge_length is the coarsest number of cells for
    which the Eikonal of every case is within 3 % of its theoretical value,
    the layered hypershape mesh is conforming, and the analytical
    approximation stays within the test tolerance for the homogeneous model
    with fitting_c = (0, 0, 0, 0), its default. The marked
    f_est (*) is the
    value with the smallest relative error abs(eik_min - eik_thr) / eik_thr,
    where eik_thr is the theoretical value, and is the one used by the test.
    In every table eik_min increases with f_est, so the smallest error is at
    one of the two listed values around eik_thr. For the heterogeneous model
    the marked f_est is that of test_eikonal.py. Every marked f_est
    converges, without retries, with Firedrake 2025.4.1 and 2026.10.0.

    Expected values. They are the fundamental frequency of the continuous
    model, so they depend neither on the mesh nor on the solver. The modal
    problem is -div(c^2 grad p) = omega^2 p, with p free (Neumann) on the
    whole boundary of the domain extended by the layer. In the rectangular
    layers c depends only on x, so the problem separates and the
    fundamental mode varies only along x. Its frequency is exact
    (rectangular_layer_frequency, checked by
    test_rectangular_expected_frequency): c / (2 (length_x + 2 pad)) in the
    homogeneous model, 0.46875 Hz in 2D and 0.47727 Hz in 3D, and the root
    of the equation in its Notes in the heterogeneous one, 0.45613 Hz in 2D
    and 0.42572 Hz in 3D.

    The hypershape layers have no closed form. Their domain is
    |x - 0.5|^n + |y - 0.5|^n + |z + 0.5|^n <= a^n (no y in 2D), cut by the
    free surface z = 0, with the degree n and the semi-axis a of the layered
    mesh of the test. Its fundamental frequency was computed with Lagrange
    elements of degree 2 on gmsh meshes of this exact geometry, with the
    boundary nodes moved onto the exact surface and, in the heterogeneous
    model, the cells split at the interface x = 0.5. The relative error of
    the mesh volume and the error of the frequency both decrease as h^2, so
    the frequencies of the two finest meshes were extrapolated linearly in
    the volume error to zero:

    Case        n    a [km]  h [km]  Volume error  f [Hz]     Expected [Hz]
    T-2D hom   2.5   0.800   0.025     -1.18e-04   0.5085727  0.50853
                             0.0125    -2.98e-05   0.5085444
    T-2D het   2.0   1.000   0.025     -8.59e-05   0.5019547  0.50193
                             0.0125    -2.15e-05   0.5019348
    T-3D hom   6.0   11/14   0.050     -7.43e-04   0.5016040  0.50139
                             0.035     -3.89e-04   0.5015015
    T-3D het   2.4   15/14   0.050     -5.72e-04   0.5084176  0.50831
                             0.035     -2.98e-04   0.5083662

    Elements of degree 3 change the frequency by less than 2e-6 Hz on the
    same meshes. The deviations of the solvers from the expected values are
    listed at the end of these Notes.

    Every case runs the nine solvers of MODAL_CONFIG. The 3D cases are
    marked slow, so the CI runs them with the slow tests. In the
    heterogeneous rectangular case, T-3D has 192964 dofs (KMV degree 3, six
    tetrahedra per cube) against 71944 for Q-3D (spectral degree 3) on the
    same grid of cubes, and the T-3D case takes about 150 s here.

    texe and pmem are the runtime and the peak memory traced by Python for
    each solver, measured serially on one core with Firedrake 2025.4.1 and
    the compilation caches already filled. PETSc allocates its memory outside
    Python, so pmem leaves out the MUMPS factorization of ARNOLDI and
    LANCZOS, the GAMG preconditioner of LOBPCG and the SLEPc solves of the
    KRYLOVSCH methods. With empty compilation caches, as in the CI, the
    Q-3D heterogeneous case took 109 s here (setup included), against
    about 27 s with filled caches: compiling the forms costs more than the
    solvers.

    Calibration of fitting_c. The ANALYTICAL solver needs the equivalent
    velocity c_eq of a static problem with Neumann conditions on the whole
    boundary, loaded at the source. `Modal_Analytical_Solver.c_equivalent`
    removes the mean of the load, since such a problem only has a solution
    for a zero-mean load, and solves it with linear elements on the mesh of
    the modal problem, which makes the ANALYTICAL solver much cheaper than
    the FEM solvers (its Notes explain both). A constant velocity needs no
    static solve, so the homogeneous model keeps c_eq = c and passes
    fitting_c = (0, 0, 0, 0), the default of the analytical solver.

    c_eq depends on the static problem, so the heterogeneous fitting_c is
    calibrated for it, separately for 2D and 3D. The values fitted to the
    zero-mean solve in the space of the modal problem, (-0.9, 0.3, -2.6, -1.4)
    in 2D and (-0.8, 0.3, -2.4, -1.7) in 3D, give errors of about -9 % in 2D
    and -11 % in 3D with linear elements. fitting_c only enters the
    regression of the equivalent velocity, after the static solves, so a
    grid with step 0.1 (fc1 and fc2 from -1 to 1, fp1 and fp2 from -5 to 3)
    was searched for the smallest maximum relative error
    abs(fana - fnum) / fnum over the three heterogeneous cases of each
    dimension. fc2 has no effect on these cases: in
    c_ref = max(c_eq * f3**fc1, c_eqref * f3**fc2), where f3 is the ratio
    of the rectangular to the elliptical frequency factor, the first term is
    the larger one, so fc2 keeps its previous value of 0.3.

    Dim   fitting_c                 Max error   Max error, one parameter +-0.1
    2D    (-0.5, 0.3, -1.5, -0.9)      0.95 %         3.97 %
    3D    (-0.9, 0.3, -2.1, -1.5)      0.73 %         3.61 %

    Both approaches for the heterogeneous models: the calibrated fitting_c
    of this test, used without optimization, and the fitting_c optimized
    for each model separately by test_modal_analytical_solver.py, with SLSQP
    from (0, 0, 0, 0). fana in Hz and its error against the expected value:

    Case       Approach    fitting_c                           fana    Error
    T-2D N2.0  calibrated  (-0.500,  0.300, -1.500, -0.900)  0.49753  -0.88 %
               optimized   ( 0.245,  0.038,  0.000, -0.551)  0.50225  +0.06 %
    T-2D REC   calibrated  (-0.500,  0.300, -1.500, -0.900)  0.45256  -0.78 %
               optimized   ( 0.561,  0.172, -0.631,  0.000)  0.45644  +0.07 %
    Q-2D REC   calibrated  (-0.500,  0.300, -1.500, -0.900)  0.45832  +0.48 %
               optimized   ( 0.056,  0.963, -0.877,  0.000)  0.45694  +0.18 %
    T-3D N2.4  calibrated  (-0.900,  0.300, -2.100, -1.500)  0.51162  +0.65 %
               optimized   ( 0.158,  0.662, -0.170, -0.998)  0.50852  +0.04 %
    T-3D REC   calibrated  (-0.900,  0.300, -2.100, -1.500)  0.42106  -1.09 %
               optimized   ( 0.162,  0.837, -1.002, -0.001)  0.42590  +0.04 %
    Q-3D REC   calibrated  (-0.900,  0.300, -2.100, -1.500)  0.41427  -2.69 %
               optimized   ( 0.163,  0.777, -0.943, -0.001)  0.42592  +0.05 %

    The homogeneous models pass (0, 0, 0, 0). The optimization keeps
    (0, 0, 0, 0) for the rectangular layers, where it already gives the
    expected value within 0.03 %. It moves T-2D N2.5 to
    (0.011, 0.010, -0.025, -0.130), and fana from 0.52430 Hz (+3.10 %) to
    0.50854 Hz (+0.00 %), and T-3D N6.0 to (0.010, 0.012, -0.101, -0.047),
    and fana from 0.51628 Hz (+2.97 %) to 0.50130 Hz (-0.02 %) (see
    test_modal_analytical_solver.py). With Firedrake
    2026.10.0 the optimized values of the 2D heterogeneous models are
    identical, and those of the 3D ones differ slightly, since the velocity
    of the cells with a centroid on x = 0.5 depends on round-off (see
    test_modal_analytical_solver.py).

    Theoretical Eikonal values. eik_min is the first-arrival time on the
    absorbing boundaries (z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1).
    The source is at (z, x) = (-0.5, 0.25) km ((-0.5, 0.25, 0.5) km in 3D),
    so the nearest absorbing boundary is x = 0, 0.25 km away, and the
    straight ray to it stays in the medium around the source. Hence
    eik_min = 0.25 / 1.5 = 166.667 ms in the homogeneous model (1.5 km/s) and
    eik_min = 0.25 / 3.0 = 83.333 ms in the heterogeneous one (3.0 km/s for
    x < 0.5).
    Every other absorbing boundary is at least 0.5 km away in the same
    medium, so its first arrival is at least twice as late.

    =======================================================
    Natural Frequency for 2D model Δx = 1000/10 m - Ele = T
    =======================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.05  163.373
     0.06  166.193*
     0.07  168.555

    *RESULTS HOMOGENEOUS
    Frequency[Hz]    N2.5      (texe/pmem)     REC      (texe/pmem)
    ANALYTICAL    0.52430 (0.217s/1.301MB) 0.46875 (0.147s/0.452MB)
    ARNOLDI       0.51046 (0.106s/5.514MB) 0.46875 (0.096s/5.977MB)
    LANCZOS       0.51046 (0.041s/5.189MB) 0.46875 (0.048s/5.629MB)
    LOBPCG        0.51046 (0.211s/6.655MB) 0.46875 (0.219s/7.207MB)
    KRYLOVSCH_CH  0.51046 (0.051s/0.139MB) 0.46875 (0.059s/0.111MB)
    KRYLOVSCH_CG  0.51046 (0.061s/0.122MB) 0.46875 (0.059s/0.120MB)
    KRYLOVSCH_GH  0.51046 (0.049s/0.138MB) 0.46875 (0.059s/0.097MB)
    KRYLOVSCH_GG  0.51046 (0.047s/0.125MB) 0.46875 (0.058s/0.133MB)
    RAYLEIGH      0.51084 (0.244s/1.151MB) 0.46875 (0.218s/1.245MB)

    *ANALYTICAL
       Case      N2.5*     REC*
    fnum[Hz]   0.51046  0.46875
    fana[Hz]   0.52430  0.46875
    fray[Hz]   0.51084  0.46875

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.05  81.498
     0.06  82.942*
     0.07  84.160

    *RESULTS HETEROGENEOUS - fitting_c = (-0.5, 0.3, -1.5, -0.9)
    Frequency[Hz]    N2.0       (texe/pmem)     REC       (texe/pmem)
    ANALYTICAL    0.49753 (0.546s/ 1.853MB) 0.45256 (0.399s/ 1.662MB)
    ARNOLDI       0.50230 (0.088s/ 7.637MB) 0.45539 (0.138s/ 8.438MB)
    LANCZOS       0.50230 (0.050s/ 7.585MB) 0.45539 (0.080s/ 8.071MB)
    LOBPCG        0.50230 (0.292s/ 9.731MB) 0.45539 (0.229s/10.373MB)
    KRYLOVSCH_CH  0.50230 (0.094s/ 0.141MB) 0.45539 (0.073s/ 0.122MB)
    KRYLOVSCH_CG  0.50230 (0.085s/ 0.098MB) 0.45539 (0.069s/ 0.132MB)
    KRYLOVSCH_GH  0.50230 (0.082s/ 0.131MB) 0.45539 (0.071s/ 0.123MB)
    KRYLOVSCH_GG  0.50230 (0.078s/ 0.138MB) 0.45539 (0.080s/ 0.110MB)
    RAYLEIGH      0.52536 (0.205s/ 1.123MB) 0.47634 (0.220s/ 1.527MB)

    *ANALYTICAL
       Case      N2.0*     REC*
    fnum[Hz]   0.50230  0.45539
    fana[Hz]   0.49753  0.45256
    fray[Hz]   0.52536  0.47634

    *RAYLEIGH N2.0
    n_eigfunc        2      *4       6       8
    freq[Hz]   0.65200 0.52536 0.51468 0.51121
    texe[s]      0.119   0.176   0.331   0.585
    mem[MB]      0.735   1.002   2.211   3.871

    =======================================================
    Natural Frequency for 2D model Δx = 1000/10 m - Ele = Q
    =======================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.04  164.457
     0.05  167.703*
     0.06  170.399

    *RESULTS HOMOGENEOUS
    Frequency[Hz]     REC      (texe/pmem)
    ANALYTICAL    0.46875 (0.189s/0.490MB)
    ARNOLDI       0.46875 (0.127s/6.100MB)
    LANCZOS       0.46875 (0.056s/5.643MB)
    LOBPCG        0.46875 (0.097s/5.836MB)
    KRYLOVSCH_CH  0.46875 (0.071s/0.166MB)
    KRYLOVSCH_CG  0.46875 (0.074s/0.128MB)
    KRYLOVSCH_GH  0.46875 (0.068s/0.121MB)
    KRYLOVSCH_GG  0.46875 (0.054s/0.125MB)
    RAYLEIGH      0.46875 (0.258s/1.360MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.46875
    fana[Hz]   0.46875
    fray[Hz]   0.46875

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.04  82.101
     0.05  83.744*
     0.06  85.118

    *RESULTS HETEROGENEOUS - fitting_c = (-0.5, 0.3, -1.5, -0.9)
    Frequency[Hz]     REC      (texe/pmem)
    ANALYTICAL    0.45832 (0.580s/2.665MB)
    ARNOLDI       0.45542 (0.118s/8.571MB)
    LANCZOS       0.45542 (0.053s/8.086MB)
    LOBPCG        0.45542 (0.092s/8.377MB)
    KRYLOVSCH_CH  0.45542 (0.067s/0.143MB)
    KRYLOVSCH_CG  0.45542 (0.073s/0.118MB)
    KRYLOVSCH_GH  0.45542 (0.068s/0.143MB)
    KRYLOVSCH_GG  0.45542 (0.075s/0.129MB)
    RAYLEIGH      0.47619 (0.255s/1.495MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.45542
    fana[Hz]   0.45832
    fray[Hz]   0.47619

    ======================================================
    Natural Frequency for 3D model Δx = 1000/7 m - Ele = T
    ======================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.04  159.016
     0.05  164.698*
     0.06  170.789

    *RESULTS HOMOGENEOUS
    Frequency[Hz]    N6.0        (texe/pmem)     REC        (texe/pmem)
    ANALYTICAL    0.51628 (0.749s/  4.654MB) 0.47741 (0.667s/  2.424MB)
    ARNOLDI       0.52345 (4.440s/193.113MB) 0.47727 (5.777s/232.052MB)
    LANCZOS       0.52345 (3.152s/191.810MB) 0.47727 (4.672s/230.583MB)
    LOBPCG        0.52345 (4.863s/242.746MB) 0.47727 (5.942s/291.832MB)
    KRYLOVSCH_CH  0.52345 (2.960s/  0.110MB) 0.47727 (4.907s/  0.098MB)
    KRYLOVSCH_CG  0.52345 (3.189s/  0.119MB) 0.47727 (4.598s/  0.123MB)
    KRYLOVSCH_GH  0.52345 (3.212s/  0.135MB) 0.47727 (4.387s/  0.109MB)
    KRYLOVSCH_GG  0.52345 (3.626s/  0.128MB) 0.47727 (4.449s/  0.104MB)
    RAYLEIGH      0.52678 (2.466s/ 36.157MB) 0.47727 (2.343s/ 43.253MB)

    *ANALYTICAL
       Case      N6.0*     REC*
    fnum[Hz]   0.52345  0.47727
    fana[Hz]   0.51628  0.47741
    fray[Hz]   0.52678  0.47727

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.04  78.971
     0.05  82.070*
     0.06  85.267

    *RESULTS HETEROGENEOUS - fitting_c = (-0.9, 0.3, -2.1, -1.5)
    Frequency[Hz]    N2.4         (texe/pmem)     REC         (texe/pmem)
    ANALYTICAL    0.51162 ( 1.199s/  3.656MB) 0.42106 ( 1.854s/  4.636MB)
    ARNOLDI       0.51202 ( 7.437s/331.678MB) 0.42207 (24.468s/524.757MB)
    LANCZOS       0.51202 ( 5.233s/330.273MB) 0.42207 (14.862s/522.047MB)
    LOBPCG        0.51202 (14.008s/417.952MB) 0.42207 (40.075s/660.653MB)
    KRYLOVSCH_CH  0.51202 ( 7.231s/  0.119MB) 0.42207 (19.994s/  0.127MB)
    KRYLOVSCH_CG  0.51202 ( 7.134s/  0.121MB) 0.42207 (16.964s/  0.122MB)
    KRYLOVSCH_GH  0.51202 ( 6.796s/  0.141MB) 0.42207 (12.236s/  0.105MB)
    KRYLOVSCH_GG  0.51202 ( 6.687s/  0.123MB) 0.42207 (11.812s/  0.134MB)
    RAYLEIGH      0.53675 ( 3.219s/ 60.774MB) 0.43878 ( 4.738s/ 95.369MB)

    *ANALYTICAL
       Case      N2.4*     REC*
    fnum[Hz]   0.51202  0.42207
    fana[Hz]   0.51162  0.42106
    fray[Hz]   0.53675  0.43878

    *RAYLEIGH N2.4
    n_eigfunc        2      *4       6
    freq[Hz]   0.61336 0.53675 0.52566
    texe[s]      2.249   3.294  11.126
    mem[MB]      9.811  60.659 206.966

    ======================================================
    Natural Frequency for 3D model Δx = 1000/7 m - Ele = Q
    ======================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 166.667 ms (Theoretical value)
    f_est  eik[ms]
     0.07  162.852
     0.08  168.726*
     0.09  174.708

    *RESULTS HOMOGENEOUS
    Frequency[Hz]     REC        (texe/pmem)
    ANALYTICAL    0.47741 (0.693s/  3.335MB)
    ARNOLDI       0.47727 (2.868s/174.051MB)
    LANCZOS       0.47727 (1.954s/172.121MB)
    LOBPCG        0.47727 (0.737s/174.172MB)
    KRYLOVSCH_CH  0.47727 (1.727s/  0.131MB)
    KRYLOVSCH_CG  0.47727 (1.892s/  0.132MB)
    KRYLOVSCH_GH  0.47727 (1.703s/  0.144MB)
    KRYLOVSCH_GG  0.47727 (1.820s/  0.119MB)
    RAYLEIGH      0.47727 (2.027s/ 19.146MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.47727
    fana[Hz]   0.47741
    fray[Hz]   0.47727

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.07  81.431
     0.08  84.377*
     0.09  87.376

    *RESULTS HETEROGENEOUS - fitting_c = (-0.9, 0.3, -2.1, -1.5)
    Frequency[Hz]     REC        (texe/pmem)
    ANALYTICAL    0.41427 (1.922s/  8.021MB)
    ARNOLDI       0.41127 (7.781s/391.781MB)
    LANCZOS       0.41127 (4.209s/389.293MB)
    LOBPCG        0.41127 (1.499s/393.876MB)
    KRYLOVSCH_CH  0.41127 (5.201s/  0.144MB)
    KRYLOVSCH_CG  0.41127 (5.315s/  0.096MB)
    KRYLOVSCH_GH  0.41127 (5.159s/  0.144MB)
    KRYLOVSCH_GG  0.41127 (5.609s/  0.122MB)
    RAYLEIGH      0.42935 (3.331s/ 38.496MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.41127
    fana[Hz]   0.41427
    fray[Hz]   0.42935

    *RAYLEIGH REC
    n_eigfunc        2      *4       6
    freq[Hz]   0.46713 0.42935 0.42125
    texe[s]      2.073   3.150  11.544
    mem[MB]      7.528  36.147 123.922

    The tables were computed with Firedrake 2025.4.1. With Firedrake
    2026.10.0 all cases pass (16 passed, 4 skipped), and eik_min agrees
    within 0.02 ms (in ms, 2026.10.0 vs 2025.4.1):

    Case     Homogeneous         Heterogeneous
    T-2D     166.187  166.193    82.949  82.942
    Q-2D     167.703  167.703    83.744  83.744
    T-3D     164.686  164.698    82.065  82.070
    Q-3D     168.744  168.726    84.375  84.377

    In the homogeneous model every solver gives the same frequency with both
    versions. In the heterogeneous model the frequencies differ as follows
    (in Hz, 2026.10.0 vs 2025.4.1; FEM is the value of all the FEM solvers,
    which agree with each other):

    Case          FEM              RAYLEIGH
    T-2D N2.0     0.50230 0.50230  0.52536 0.52536
    T-2D REC      0.45539 0.45539  0.47634 0.47634
    Q-2D REC      0.45542 0.45542  0.47619 0.47619
    T-3D N2.4     0.52573 0.51202  0.55169 0.53675
    T-3D REC      0.43493 0.42207  0.45261 0.43878
    Q-3D REC      0.42883 0.41127  0.44576 0.42935

    The velocity has one value per cell, evaluated at the cell centroid,
    and on the 1/7 km meshes some centroids lie exactly on the interface
    x = 0.5. Whether such a cell takes the fast or the slow velocity depends
    on round-off, which differs between the versions. In 2D (1/10 km) no
    centroid lies on x = 0.5, and the values are identical.

    Deviation from the expected value with 2026.10.0 (2025.4.1) and the
    tolerance of the test. In the homogeneous models the frequencies are
    the same with both versions, and those of the rectangular layers equal
    the expected values:

    Case          FEM (5 %)           RAYLEIGH (9 %)
    T-2D N2.5     +0.38 % (+0.38 %)   +0.45 % (+0.45 %)
    T-3D N6.0     +4.40 % (+4.40 %)   +5.06 % (+5.06 %)
    T-2D N2.0     +0.07 % (+0.07 %)   +4.67 % (+4.67 %)
    T-2D REC      -0.16 % (-0.16 %)   +4.43 % (+4.43 %)
    Q-2D REC      -0.16 % (-0.16 %)   +4.40 % (+4.40 %)
    T-3D N2.4     +3.43 % (+0.73 %)   +8.53 % (+5.60 %)
    T-3D REC      +2.16 % (-0.86 %)   +6.32 % (+3.07 %)
    Q-3D REC      +0.73 % (-3.39 %)   +4.71 % (+0.85 %)

    The tolerances cover both versions. The smallest margins are 0.60 % for
    FEM in T-3D N6.0, whose layered mesh of 1/7 km cells approximates the
    hypershape of degree 6 less closely than the other layers, and 0.47 %
    for RAYLEIGH in T-3D N2.4 with 2026.10.0.

    The ANALYTICAL frequencies of the homogeneous models and of the 2D
    heterogeneous models are identical with both versions. In 3D the
    heterogeneous c_eq changes with the velocity of the cells on x = 0.5, as
    FEM does (in Hz, 2026.10.0 vs 2025.4.1, and deviation from the expected
    value, tolerance 9 %):

    Case          ANALYTICAL       Deviation
    T-3D N2.4     0.52895 0.51162  +4.06 % (+0.65 %)
    T-3D REC      0.43428 0.42106  +2.01 % (-1.09 %)
    Q-3D REC      0.42305 0.41427  -0.63 % (-2.69 %)

    Against the FEM frequency of its own version, ANALYTICAL is within
    1.35 % with 2026.10.0 and 0.95 % with 2025.4.1. Before the zero-mean
    load, the 2D ANALYTICAL frequencies differed by up to 5.8 % between the
    versions.

    With 2026.10.0, when the 3D cases still ran fewer solvers, those of Q-3D
    REC heterogeneous took 24.3 s in total (ARNOLDI 6.0 s, LANCZOS 3.8 s,
    LOBPCG 1.2 s), and those of T-3D REC heterogeneous 27.9 s (KRYLOVSCH and
    RAYLEIGH). LOBPCG reached its tolerance in every case.
    """

    case = (dimension, element_geometry, homogeneous, layer_shape)
    degree_layer = case_parameter(MODAL_CONFIG["degree_layer"], case)
    c_hom = "Homogeneous" if homogeneous else "Heterogeneous"
    n_hyp = f"HyperShape N{degree_layer}" if degree_layer is not None else "Rectangular"

    SpyroEnsemble.print(
        "\n"
        + 60 * "="
        + "\nTesting Modal Solvers with "
        + f"{element_geometry} elements for {dimension}D case\nand "
        + f"{n_hyp} layer. Propagation Speed: {c_hom}\n"
        + 60 * "="
    )

    # ============ SIMULATION PARAMETERS ============

    wave, fitting_c, modal_solver_lst = wave_instance

    # ============ EXPECTED VALUES ============

    exp_value = case_parameter(MODAL_CONFIG["exp_value"], case)

    try:
        # Computing the fundamental frequency
        run_modal(wave, modal_solver_lst, fitting_c, exp_value)

        # Renaming the folder if degree_layer is modified
        wave.layer_ops.rename_folder_habc()

    except ConvergenceError as e:
        fail(
            f"Checking Modal Solvers with {element_geometry} elements "
            f"for {dimension}D case, {n_hyp} layer and {c_hom} propagation "
            f"speed raised an exception: {str(e)}"
        )
