"""Unit tests for the Modal solvers implemented in spyro.solvers.modal.modal_sol.

These tests verify the implemented modal solvers by comparing the computed fundamental
frequency with expected values for different domain configurations. The tests cover
both 2D and 3D cases, with homogeneous and heterogeneous velocity profiles.
"""

from pytest import fail, fixture, mark, param
from firedrake import conditional, ConvergenceError
from firedrake import COMM_WORLD as comm
from numpy import isclose
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
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Hom" if homogeneous else "Het")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/modal_test{dimension}d/modal_test{dimension}d" + str_ele
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
    Wave_obj : acoustic_wave.AcousticWave
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
    Wave_obj = AcousticWave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    if homogeneous:
        Wave_obj.set_initial_velocity_model(constant=1.5)

    else:
        cond = conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
        Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.mesh_ops.preamble_mesh_operations(Wave_obj, f_est=f_est)

    # ============ EIKONAL ANALYSIS ============

    # Finding critical points
    Wave_obj.layer_ops.critical_boundary_points(Wave_obj)

    # ============ MODAL ANALYSIS ============

    # Modal solvers
    if dimension == 2:
        modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
                            'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG', 'RAYLEIGH']

    if dimension == 3:
        if element_geometry == "T":
            modal_solver_lst = ['ANALYTICAL', 'KRYLOVSCH_CH', 'KRYLOVSCH_GH', 'RAYLEIGH']
        else:
            modal_solver_lst = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG',
                                'KRYLOVSCH_CG', 'KRYLOVSCH_GG', 'RAYLEIGH']

    return Wave_obj, fitting_c, modal_solver_lst


def run_modal(Wave_obj, modal_solver_lst, fitting_c, exp_value, n_root=1):
    """
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022).

    Parameters
    ----------
    Wave_obj : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    modal_solver_lst : `list`
        List of methods to be used to solve the eigenvalue problem.
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS',
        'LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
        'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
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
    Wave_obj.layer_ops.layer_size_criterion(Wave_obj.mesh_parameters.lmin, n_root=n_root)

    # Creating mesh with absorbing layer
    Wave_obj.layer_ops.create_mesh_with_layer(Wave_obj)

    # Updating velocity model
    Wave_obj.layer_ops.velocity_abc(Wave_obj)

    # Loop for different modal solvers
    for modal_solver in modal_solver_lst:

        # Modal solver
        pprint(f"\nModal Solver: {modal_solver}", comm=comm)

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Computing fundamental frequency
        Wave_obj.layer_ops.fundamental_frequency(Wave_obj, method=modal_solver,
                                                 fitting_c=fitting_c)

        # Estimating computational resource usage
        name_cost = Wave_obj.path_case_abc + modal_solver + "_"
        comp_cost("tfin", tRef=tRef, user_name=name_cost)

        tol = 0.07 if (modal_solver == 'ANALYTICAL'
                       or modal_solver == 'RAYLEIGH') else 0.05

        abc_str = Wave_obj.case_abc if Wave_obj.layer_ops.layer_geometry.n_hyp is None \
            else f"{Wave_obj.case_abc[:2]}" + \
            f"{Wave_obj.layer_ops.layer_geometry.n_hyp:.1f}{Wave_obj.case_abc[-4:]}"
        met_str = f"Fundamental Frequency {abc_str} {Wave_obj.dimension}D. "
        met_str += f"Method {modal_solver}"
        cmp_str = f"Expected {exp_value:.5f}, got = {Wave_obj.fundam_freq:.5f}"
        assert isclose(Wave_obj.fundam_freq / exp_value, 1., atol=tol), \
            "✗ " + met_str + "  → " + cmp_str
        pprint("✓ " + met_str + " Verified: " + cmp_str, comm=comm)


@mark.older_firedrake
@mark.parametrize("element_geometry, dimension, degree_layer, homogeneous",
                  [("T", 2, 2.5, True),
                   ("T", 2, None, True),
                   ("T", 2, 2.0, False),
                   ("T", 2, None, False),
                   param("T", 3, None, True, marks=mark.slow),
                   param("T", 3, 6.0, True, marks=mark.slow),
                   param("T", 3, 2.4, False, marks=mark.slow),
                   param("T", 3, None, False, marks=mark.slow),
                   param("Q", 3, None, True, marks=mark.slow),
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

    *RESULTS HOMOGENEOUS
    Frequency[Hz]    N2.5      (texe/pmem)     REC      (texe/pmem)
    ANALYTICAL    0.50934 (0.359s/2.160MB) 0.46875 (0.352s/2.146MB)
    ARNOLDI       0.51046 (0.078s/4.665MB) 0.46875 (0.104s/5.066MB)
    LANCZOS       0.51046 (0.047s/4.102MB) 0.46875 (0.044s/4.435MB)
    LOBPCG        0.51046 (1.983s/4.025MB) 0.46875 (1.543s/4.359MB)
    KRYLOVSCH_CH  0.51046 (0.045s/0.084MB) 0.46875 (0.026s/0.084MB)
    KRYLOVSCH_CG  0.51046 (0.038s/0.070MB) 0.46875 (0.036s/0.071MB)
    KRYLOVSCH_GH  0.51046 (0.071s/0.078MB) 0.46875 (0.031s/0.078MB)
    KRYLOVSCH_GG  0.51046 (0.068s/0.091MB) 0.46875 (0.027s/0.091MB)
    RAYLEIGH      0.52572 (1.764s/3.309MB) 0.48624 (1.579s/3.315MB)

    *ANALYTICAL
       Case      REC*   N2.5*
    fnum[Hz]  0.46875 0.50934
    fana[Hz]  0.46875 0.51046
    fray[Hz]  0.48624 0.52572

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

    *RESULTS HETEROGENEOUS
    Frequency[Hz]    N2.0      (texe/pmem)     REC      (texe/pmem)
    ANALYTICAL    0.50428 (0.462s/3.152MB) 0.45737 (0.652s/2.990MB)
    ARNOLDI       0.50440 (0.102s/6.693MB) 0.45539 (0.128s/7.118MB)
    LANCZOS       0.50440 (0.086s/5.965MB) 0.45539 (0.071s/6.350MB)
    LOBPCG        0.50440 (4.269s/5.898MB) 0.45539 (3.752s/6.212MB)
    KRYLOVSCH_CH  0.50440 (0.038s/0.085MB) 0.45539 (0.047s/0.083MB)
    KRYLOVSCH_CG  0.50440 (0.040s/0.072MB) 0.45539 (0.044s/0.072MB)
    KRYLOVSCH_GH  0.50440 (0.048s/0.077MB) 0.45539 (0.042s/0.078MB)
    KRYLOVSCH_GG  0.50440 (0.054s/0.097MB) 0.45539 (0.039s/0.095MB)
    RAYLEIGH      0.52783 (1.512s/3.667MB) 0.47634 (1.467s/3.672MB)

    *ANALYTICAL
       Case      REC*   N2.0*
    fnum[Hz]  0.45737 0.50428
    fana[Hz]  0.45503 0.50807
    fray[Hz]  0.47634 0.52783

    *RAYLEIGH N2.0
    n_eigfunc       2      *4       6       8
    freq[Hz]  0.66237 0.52783 0.51705 0.51355
    texe[s]     0.263   1.956   5.947  17.152
    mem[MB]     1.359   3.792   8.075  13.311

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = T
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.02  146.002*
     0.03  153.839

    *RESULTS HOMOGENEOUS
    Frequency[Hz]    N5.5        (texe/pmem)     REC        (texe/pmem)
    ANALYTICAL    0.51628 ( 2.918s/ 5.911MB) 0.47727 ( 3.259s/ 5.725MB)
    KRYLOVSCH_CH  0.52345 ( 9.954s/ 0.936MB) 0.47727 (14.255s/ 0.925MB)
    KRYLOVSCH_GH  0.52345 ( 9.869s/ 0.077MB) 0.47727 (14.509s/ 0.075MB)
    RAYLEIGH      0.55066 (29.679s/48.528MB) 0.49966 (32.511s/55.762MB)

    *EIKONAL HETEROGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.03 76.777
     0.04 79.409
     0.05 82.273*
     0.06 85.347

    *RESULTS HETEROGENEOUS
    Frequency[Hz]    N2.4        (texe/pmem)     REC         (texe/pmem)
    ANALYTICAL    0.51833 ( 5.761s/10.364MB) 0.42415 ( 8.884s/ 12.748MB)
    KRYLOVSCH_CH  0.51535 (24.633s/ 0.935MB) 0.42562 (66.466s/  0.926MB)
    KRYLOVSCH_GH  0.51535 (25.103s/ 0.077MB) 0.42562 (64.295s/  0.075MB)
    RAYLEIGH      0.54617 (38.478s/73.414MB) 0.44942 (51.352s/108.206MB)

    ANALYTICAL
       Case      REC*  N2.4*
    fnum[Hz]  0.42136 0.51833
    fana[Hz]  0.42562 0.51535
    fray[Hz]  0.44942 0.54617

    RAYLEIGH N2.4
    n_eigfunc       2      *4       6
    freq[Hz]  0.65356 0.54617 0.53122
    texe[s]     0.799  34.327 373.401
    mem[MB]     6.730  47.889 154.636

    ===================================================
    Natural Frequency for 3D model Δx = 150m - Ele = Q
    ===================================================

    *EIKONAL HOMOGENEOUS
    eik_min = 83.333 ms
    f_est  eik[ms]
     0.02  138.931*
     0.02  142.020

    *RESULTS HETEROGENEOUS
    Frequency[Hz]     REC         (texe/pmem)
    ANALYTICAL    0.47741 ( 2.239s/  9.275MB)
    ARNOLDI       0.47727 ( 8.990s/145.647MB)
    LANCZOS       0.47727 ( 7.876s/ 96.980MB)
    LOBPCG        0.47727 ( 7.454s/ 95.550MB)
    KRYLOVSCH_CG  0.47727 ( 5.036s/  0.084MB)
    KRYLOVSCH_GG  0.47727 ( 4.860s/  0.076MB)
    RAYLEIGH      0.49966 (30.048s/ 33.385MB)

        Case      REC*
    fnum[Hz]  0.47741
    fana[Hz]  0.47727
    fray[Hz]  0.49966

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

    *RESULTS HETEROGENEOUS
    Frequency[Hz]     REC          (texe/pmem)
    ANALYTICAL    0.41373 ( 4.707s/ 11.191MB)
    ARNOLDI       0.41127 (32.395s/326.702MB)
    LANCZOS       0.41127 (31.732s/218.844MB)
    LOBPCG        0.41127 (35.811s/215.802MB)
    KRYLOVSCH_CG  0.41127 (25.506s/  0.085MB)
    KRYLOVSCH_GG  0.41127 (25.221s/  0.086MB)
    RAYLEIGH      0.43304 (29.833s/ 53.146MB)

    ANALYTICAL
       Case      REC*
    fnum[Hz]  0.41127
    fana[Hz]  0.41373
    fray[Hz]  0.43304

    RAYLEIGH REC
    n_eigfunc       2      *4       6
    freq[Hz]  0.50637 0.43304 0.42081
    texe[s]     0.859  25.615 497.458
    mem[MB]     8.168  51.299 185.377
    """

    c_hom = "Homogeneous" if homogeneous else "Heterogeneous"
    n_hyp = f"HyperShape N{degree_layer}" if degree_layer is not None else "Rectangular"

    pprint("\n" + 60 * "=" + f"\nTesting Modal Solvers with {element_geometry} elements"
           + f"for {dimension}D case\nand {n_hyp} layer. Propagation Speed: {c_hom}\n"
           + 60 * "=", comm=comm)

    # ============ SIMULATION PARAMETERS ============

    Wave_obj, fitting_c, modal_solver_lst = wave_instance

    # ============ EXPECTED VALUES ============

    if dimension == 2:
        if homogeneous:
            exp_value = 0.46875 if Wave_obj.abc_deg_layer is None else 0.51046

        else:
            exp_value = 0.45539 if Wave_obj.abc_deg_layer is None else 0.50440

    if dimension == 3:
        if element_geometry == "T":
            if homogeneous:
                exp_value = 0.47727 if Wave_obj.abc_deg_layer is None else 0.52345
            else:
                exp_value = 0.42562 if Wave_obj.abc_deg_layer is None else 0.51535
        else:
            if homogeneous:
                exp_value = 0.47727
            else:
                exp_value = 0.41127

    try:
        # Computing the fundamental frequency
        run_modal(Wave_obj, modal_solver_lst, fitting_c, exp_value)

        # Renaming the folder if degree_layer is modified
        Wave_obj.layer_ops.rename_folder_habc()

    except ConvergenceError as e:
        fail(f"Checking Modal Solvers with {element_geometry} elements "
             f"for{dimension}D case, {n_hyp} layer and {c_hom} propagation "
             f"speed raised an exception: {str(e)}")
