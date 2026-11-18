"""Unit tests for the Modal solvers implemented in spyro.solvers.modal.modal_sol.

These tests verify the implemented modal solvers by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from pytest import fail, fixture, mark, param
from firedrake import conditional, ConvergenceError
from numpy import isclose
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost


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
        "output_folder": f"output/modal_test{dimension}d/modal_test{dimension}d"
        + str_ele
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
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 1.0 / 10 if dimension == 2 else 1.0 / 7

    # f_est: Factor for the stabilizing term in Eikonal equation
    # fitting_c: Parameters for fitting equivalent velocity regression
    if dimension == 2:
        if element_geometry == "T":
            f_est = 0.06 if homogeneous else 0.06

        else:
            f_est = 0.05 if homogeneous else 0.05
        fitting_c = (0.0, 0.0, 0.0, 0.0) if homogeneous else (0.5, 0.3, -2.6, -1.4)

    if dimension == 3:
        if element_geometry == "T":
            f_est = 0.05 if homogeneous else 0.05

        else:
            f_est = 0.08 if homogeneous else 0.08
        fitting_c = (0.0, 0.0, 0.0, 0.0) if homogeneous else (0.5, 0.3, -1.5, -1.0)

    # Layer shape
    layer_shape = "rectangular" if degree_layer is None else "hypershape"

    # Get simulation parameters
    SpyroEnsemble.print(f"\nMesh Size: {1e3 * edge_length:.4f} m")
    SpyroEnsemble.print(f"Element Geometry: {element_geometry}")
    SpyroEnsemble.print(f"Eikonal Stabilizing Factor: {f_est:.2f}")
    SpyroEnsemble.print(f"Layer Shape: {layer_shape}")
    fit_str = "Fitting Parameters for Analytical Solver: " + 3 * "{:.1f}, "
    SpyroEnsemble.print((fit_str + "{:.1f}\n").format(*fitting_c))

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

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    if dimension == 2:
        modal_solver_lst = [
            "ANALYTICAL",
            "ARNOLDI",
            "LANCZOS",
            "LOBPCG",
            "KRYLOVSCH_CH",
            "KRYLOVSCH_CG",
            "KRYLOVSCH_GH",
            "KRYLOVSCH_GG",
            "RAYLEIGH",
        ]

    if dimension == 3:
        if element_geometry == "T":
            modal_solver_lst = [
                "ANALYTICAL",
                "KRYLOVSCH_CH",
                "KRYLOVSCH_GH",
                "RAYLEIGH",
            ]
        else:
            modal_solver_lst = [
                "ANALYTICAL",
                "ARNOLDI",
                "LANCZOS",
                "LOBPCG",
                "KRYLOVSCH_CG",
                "KRYLOVSCH_GG",
                "RAYLEIGH",
            ]

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
            wave,
            method=modal_solver,
            fitting_c=fitting_c,
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


# @mark.older_firedrake
@mark.parametrize(
    "element_geometry, dimension, degree_layer, homogeneous",
    [
        ("T", 2, 2.5, True),
        ("T", 2, None, True),
        ("T", 2, 2.0, False),
        ("T", 2, None, False),
        ("Q", 2, None, True),
        ("Q", 2, None, False),
        param("T", 3, 6.0, True, marks=mark.slow),
        param("T", 3, None, True, marks=mark.slow),
        param("T", 3, 2.4, False, marks=mark.slow),
        param("T", 3, None, False, marks=mark.slow),
        param("Q", 3, None, True, marks=mark.slow),
        param("Q", 3, None, False, marks=mark.slow),
    ],
)
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

    Parameters selected with the working-tree code (Firedrake 2025.4.1, serial):
    the coarsest mesh edge_length = 1 / N for which the Eikonal of every case
    is within 3 % of its theoretical value with a stable stabilizing factor
    (f_est - 0.01, f_est and f_est + 0.01 converge and increase smoothly), the
    layered hypershape mesh is conforming, and the analytical approximation
    stays within the test tolerance with fitting_c = (0, 0, 0, 0) for the
    homogeneous model. The marked f_est (*) is the one used by the test. The
    expected values are the fnum of the FEM solvers; for the homogeneous
    rectangular layers they equal c / (2 (length_x + 2 pad)).

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
    ANALYTICAL    0.52430 (0.478s/2.540MB) 0.46875 (0.514s/1.810MB)
    ARNOLDI       0.51046 (0.093s/5.454MB) 0.46875 (0.088s/5.927MB)
    LANCZOS       0.51046 (0.053s/5.201MB) 0.46875 (0.040s/5.641MB)
    LOBPCG        0.51046 (2.875s/5.200MB) 0.46875 (2.334s/5.641MB)
    KRYLOVSCH_CH  0.51046 (0.042s/0.113MB) 0.46875 (0.047s/0.111MB)
    KRYLOVSCH_CG  0.51046 (0.039s/0.122MB) 0.46875 (0.046s/0.122MB)
    KRYLOVSCH_GH  0.51046 (0.046s/0.126MB) 0.46875 (0.043s/0.112MB)
    KRYLOVSCH_GG  0.51046 (0.039s/0.100MB) 0.46875 (0.042s/0.126MB)
    RAYLEIGH      0.51084 (1.603s/2.794MB) 0.46875 (1.802s/2.960MB)

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

    *RESULTS HETEROGENEOUS - fitting_c = (0.5, 0.3, -2.6, -1.4)
    Frequency[Hz]    N2.0      (texe/pmem)     REC      (texe/pmem)
    ANALYTICAL    0.49913 (0.371s/1.168MB) 0.45788 (0.767s/2.102MB)
    ARNOLDI       0.50440 (0.062s/7.586MB) 0.45539 (0.141s/8.368MB)
    LANCZOS       0.50440 (0.063s/7.585MB) 0.45539 (0.102s/8.096MB)
    LOBPCG        0.50440 (5.929s/7.610MB) 0.45539 (6.495s/8.096MB)
    KRYLOVSCH_CH  0.50440 (0.055s/0.126MB) 0.45539 (0.073s/0.106MB)
    KRYLOVSCH_CG  0.50440 (0.054s/0.121MB) 0.45539 (0.077s/0.121MB)
    KRYLOVSCH_GH  0.50440 (0.052s/0.130MB) 0.45539 (0.072s/0.126MB)
    KRYLOVSCH_GG  0.50440 (0.061s/0.124MB) 0.45539 (0.083s/0.100MB)
    RAYLEIGH      0.52768 (1.782s/2.246MB) 0.47634 (2.191s/3.183MB)

    *ANALYTICAL
       Case      N2.0*     REC*
    fnum[Hz]   0.50440  0.45539
    fana[Hz]   0.49913  0.45788
    fray[Hz]   0.52768  0.47634

    *RAYLEIGH N2.0
    n_eigfunc        2      *4       6       8
    freq[Hz]   0.65545 0.52768 0.51695 0.51347
    texe[s]      0.261   1.613   8.042  22.043
    mem[MB]      1.174   2.402   5.618   8.952

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
    ANALYTICAL    0.46876 (0.426s/2.361MB)
    ARNOLDI       0.46875 (0.106s/6.039MB)
    LANCZOS       0.46875 (0.052s/5.629MB)
    LOBPCG        0.46875 (1.240s/5.639MB)
    KRYLOVSCH_CH  0.46875 (0.061s/0.130MB)
    KRYLOVSCH_CG  0.46875 (0.055s/0.123MB)
    KRYLOVSCH_GH  0.46875 (0.054s/0.113MB)
    KRYLOVSCH_GG  0.46875 (0.059s/0.102MB)
    RAYLEIGH      0.46875 (2.644s/2.860MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.46875
    fana[Hz]   0.46876
    fray[Hz]   0.46875

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.04  82.101
     0.05  83.744*
     0.06  85.118

    *RESULTS HETEROGENEOUS - fitting_c = (0.5, 0.3, -2.6, -1.4)
    Frequency[Hz]     REC      (texe/pmem)
    ANALYTICAL    0.47244 (0.465s/2.455MB)
    ARNOLDI       0.45542 (0.141s/8.513MB)
    LANCZOS       0.45542 (0.072s/8.098MB)
    LOBPCG        0.45542 (2.475s/8.096MB)
    KRYLOVSCH_CH  0.45542 (0.066s/0.142MB)
    KRYLOVSCH_CG  0.45542 (0.066s/0.111MB)
    KRYLOVSCH_GH  0.45542 (0.069s/0.143MB)
    KRYLOVSCH_GG  0.45542 (0.065s/0.117MB)
    RAYLEIGH      0.47619 (2.382s/3.141MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.45542
    fana[Hz]   0.47244
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
    ANALYTICAL    0.51628 ( 3.160s/ 7.646MB) 0.47741 ( 2.928s/ 6.556MB)
    KRYLOVSCH_CH  0.52345 (12.059s/ 0.881MB) 0.47727 (17.431s/ 0.880MB)
    KRYLOVSCH_GH  0.52345 (10.802s/ 0.108MB) 0.47727 (17.802s/ 0.107MB)
    RAYLEIGH      0.52678 (34.111s/43.835MB) 0.47727 (43.107s/53.147MB)

    *ANALYTICAL
       Case      N6.0*     REC*
    fnum[Hz]   0.52345  0.47727
    fana[Hz]   0.51628  0.47741
    fray[Hz]   0.52678  0.47727

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms (Theoretical value)
    f_est  eik[ms]
     0.04  79.370
     0.05  82.259*
     0.06  85.344

    *RESULTS HETEROGENEOUS - fitting_c = (0.5, 0.3, -1.5, -1.0)
    Frequency[Hz]    N2.4         (texe/pmem)     REC         (texe/pmem)
    ANALYTICAL    0.52358 ( 4.491s/  6.204MB) 0.41761 ( 6.882s/  9.506MB)
    KRYLOVSCH_CH  0.51535 (29.458s/  0.090MB) 0.42562 (74.857s/  0.904MB)
    KRYLOVSCH_GH  0.51535 (28.739s/  0.121MB) 0.42562 (74.787s/  0.132MB)
    RAYLEIGH      0.54073 (44.980s/ 69.687MB) 0.44257 (57.528s/105.078MB)

    *ANALYTICAL
       Case      N2.4*     REC*
    fnum[Hz]   0.51535  0.42562
    fana[Hz]   0.52358  0.41761
    fray[Hz]   0.54073  0.44257

    *RAYLEIGH N2.4
    n_eigfunc        2      *4       6
    freq[Hz]   0.61849 0.54073 0.52941
    texe[s]      0.938  46.316 511.660
    mem[MB]      9.186  68.240 221.512

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
    Frequency[Hz]     REC         (texe/pmem)
    ANALYTICAL    0.47741 ( 2.001s/  7.875MB)
    ARNOLDI       0.47727 ( 8.812s/173.767MB)
    LANCZOS       0.47727 ( 8.421s/172.121MB)
    LOBPCG        0.47727 ( 8.266s/172.119MB)
    KRYLOVSCH_CG  0.47727 ( 5.494s/  0.143MB)
    KRYLOVSCH_GG  0.47727 ( 5.359s/  0.131MB)
    RAYLEIGH      0.47727 (39.245s/ 30.717MB)

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

    *RESULTS HETEROGENEOUS - fitting_c = (0.5, 0.3, -1.5, -1.0)
    Frequency[Hz]     REC         (texe/pmem)
    ANALYTICAL    0.40927 ( 3.607s/  9.316MB)
    ARNOLDI       0.41127 (32.552s/390.963MB)
    LANCZOS       0.41127 (31.941s/389.293MB)
    LOBPCG        0.41127 (37.387s/389.291MB)
    KRYLOVSCH_CG  0.41127 (26.247s/  0.144MB)
    KRYLOVSCH_GG  0.41127 (25.564s/  0.113MB)
    RAYLEIGH      0.42935 (40.895s/ 49.867MB)

    *ANALYTICAL
       Case       REC*
    fnum[Hz]   0.41127
    fana[Hz]   0.40927
    fray[Hz]   0.42935

    *RAYLEIGH REC
    n_eigfunc        2      *4       6
    freq[Hz]   0.46713 0.42935 0.42125
    texe[s]      0.899  39.138 438.810
    mem[MB]      7.403  47.018 146.401
    """

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

    if dimension == 2:
        if element_geometry == "T":
            if homogeneous:
                exp_value = 0.46875 if wave.abc_deg_layer is None else 0.51046
            else:
                exp_value = 0.45539 if wave.abc_deg_layer is None else 0.50440
        else:
            if homogeneous:
                exp_value = 0.46875
            else:
                exp_value = 0.45542

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
        run_modal(wave, modal_solver_lst, fitting_c, exp_value)

        # Renaming the folder if degree_layer is modified
        wave.layer_ops.rename_folder_habc()

    except ConvergenceError as e:
        fail(
            f"Checking Modal Solvers with {element_geometry} elements "
            f"for {dimension}D case, {n_hyp} layer and {c_hom} propagation "
            f"speed raised an exception: {str(e)}"
        )
