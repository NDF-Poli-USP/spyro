"""Unit tests for the Modal solvers implemented in spyro.solvers.modal.modal_sol.

These tests verify the implemented modal solvers by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from pytest import fail, fixture, mark, param, skip
from firedrake import conditional, ConvergenceError
from numpy import isclose
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

        tol = (
            0.09
            if (modal_solver == "ANALYTICAL" or modal_solver == "RAYLEIGH")
            else (0.05 if wave.mesh_parameters.quadrilateral else 0.04)
        )

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
    boundaries (z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1). The source
    is at (z, x) = (-0.5, 0.25) km ((z, x, y) = (-0.5, 0.25, 0.5) km in 3D), so
    the nearest absorbing boundary is x = 0, 0.25 km away. Hence
    eik_min = 0.25 / 1.5 = 166.667 ms in the homogeneous model (1.5 km/s) and
    eik_min = 0.25 / 3.0 = 83.333 ms in the heterogeneous one (3.0 km/s for
    x < 0.5, 1.5 km/s otherwise). Every other boundary is at least twice as far
    in time.

    Selection of the parameters. The Eikonal is solved as in test_eikonal.py
    ('nearest_node', consistent elements, p = 2) with edge_length = 1/10 km in
    2D and 1/7 km in 3D. N = 1 / edge_length is the coarsest number of cells
    for which every Eikonal is within 3 % of its theoretical value, the layered
    hypershape mesh is conforming and the analytical approximation stays within
    the tolerance. The marked f_est is the one with the smallest relative error
    abs(eik_min - eik_thr) / eik_thr, and it converges without retries with
    Firedrake 2025.4.1 and 2026.10.0. The expected values are the fnum of the
    FEM solvers.

    Cost. Every case runs the nine solvers of MODAL_CONFIG. The 3D cases are
    marked slow. Compiling the forms costs more than the solvers: the Q-3D
    heterogeneous case takes 109 s with empty caches and about 27 s with filled
    caches. pmem (peak memory traced by Python) leaves out the PETSc
    allocations.

    Calibration of fitting_c. The ANALYTICAL solver needs the equivalent
    velocity c_eq of a static zero-mean Neumann problem, solved with linear
    elements. A constant velocity needs no static solve, so the homogeneous
    models use fitting_c = (0, 0, 0, 0), the default. For the heterogeneous
    models fitting_c is calibrated over a grid (step 0.1) for the smallest
    maximum relative error abs(fana - fnum) / fnum, separately in 2D and 3D. fc2
    has no effect on these cases and keeps the value 0.3.

    Dim   fitting_c                 Max error   Max error, one parameter +-0.1
    2D    (-0.5, 0.3, -1.5, -0.9)      0.64 %         3.32 %
    3D    (-0.9, 0.3, -2.1, -1.5)      0.73 %         3.80 %

    Heterogeneous results with Firedrake 2025.4.1 (frequencies in Hz). All the
    FEM solvers give fnum. fana is ANALYTICAL with the calibrated fitting_c and
    fray is RAYLEIGH. In the homogeneous models every solver agrees with fnum,
    except N2.5 (T-2D), where fana = 0.52430 and fnum = 0.51046.

    Case        f_est   fnum      fana      fray
    T-2D N2.0   0.06    0.50440   0.50526   0.52768
    T-2D REC    0.06    0.45539   0.45256   0.47634
    Q-2D REC    0.05    0.45542   0.45832   0.47619
    T-3D N2.4   0.05    0.51535   0.51669   0.54073
    T-3D REC    0.05    0.42562   0.42541   0.44257
    Q-3D REC    0.08    0.41127   0.41427   0.42935

    Firedrake versions. With Firedrake 2026.10.0 all cases pass and eik_min
    agrees within 0.02 ms. The 2D frequencies are identical. In the 3D
    heterogeneous models the frequencies change (FEM deviates from the
    expected value by up to +4.27 %, RAYLEIGH by up to +8.39 %, ANALYTICAL by
    up to +3.35 %): the velocity is evaluated at the cell centroid, and on the
    1/7 km meshes some centroids lie exactly on the interface x = 0.5, where
    the fast or slow velocity is decided by round-off. The tolerances (4 % T
    and 5 % Q for FEM, 9 % for RAYLEIGH and ANALYTICAL) cover both versions;
    the smallest margins are 0.73 % (FEM) and 0.61 % (RAYLEIGH) for Q-3D REC.

    Complete documentation, with the Eikonal, frequency, runtime and memory
    tables of every case and the Firedrake comparison:
    tests/documentation_tests/test_modal_solvers__test_modal.md
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
