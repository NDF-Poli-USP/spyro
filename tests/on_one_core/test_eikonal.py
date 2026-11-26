"""Unit tests for the Nonlinear Eikonal analysis for 2D and 3D cases.

The test compares the minimum Eikonal value obtained from the simulation with the
theoretical value for different mesh sizes, element geometries, and finite element types.
The results are expected to be within a specified tolerance of the theoretical value.
"""
from pytest import fail, mark, param
from firedrake import conditional, ConvergenceError
from numpy import isclose
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost

pytestmark = mark.eikonal

# Simulation parameters, calibrated for each source condition (see Notes of
# test_eikonal). Each parameter is a decision tree over the case fields
# (source_bc, dimension, element_type, element_geometry): every nested
# dictionary splits on exactly one field, keyed by that field's options
# (currently two each), and each leaf is the value shared by all the cases
# below it. A field is omitted where all its options share the same value, so
# that value is set once for the whole subtree.
EIKONAL_CONFIG = {
    # Mesh size in km: edge_length = lba / cpw, where cpw is the number of cells
    # per wavelength and lba = minimum_velocity / source_frequency
    "edge_length": {2: 0.1, 3: 0.15},
    # Finite element order for the Eikonal equation
    "p_eik": {"T": 2, "Q": {"consistent": 2, "underintegrated": {2: 4, 3: 3}}},
    # Absolute tolerance of eik_min / eik_thr with respect to 1
    "atol": {
        "nearest_node": {
            2: {"consistent": 5e-3, "underintegrated": 1.5e-2},
            3: {"consistent": 3e-2, "underintegrated": 2.5e-2},
        },
        "local_time": {2: 5e-3, 3: 3e-2},
    },
    # Factor for the stabilizing term in Eikonal equation
    "f_est": {
        "nearest_node": {
            "T": {"consistent": {2: 0.06, 3: 0.05}, "underintegrated": 0.07},
            "Q": {
                "consistent": {2: 0.05, 3: 0.08},
                "underintegrated": {2: 0.03, 3: 0.06},
            },
        },
        "local_time": {
            "T": {
                "consistent": {2: 0.12, 3: 0.04},
                "underintegrated": {2: 0.08, 3: 0.04},
            },
            "Q": {"consistent": {2: 0.12, 3: 0.02}, "underintegrated": 0.03},
        },
    },
}


def case_parameter(tree: dict | float | int, case: tuple) -> float | int:
    """Get the value of a parameter of EIKONAL_CONFIG for one case.

    Parameters
    ----------
    tree : `dict`, `float` or `int`
        Decision tree of the parameter. Each nested dictionary splits on exactly
        one case field and is keyed by that field's options, and each leaf is
        the value shared by all the cases below it.
    case : `tuple`
        Options of the case: (source_bc, dimension, element_type,
        element_geometry). Every option belongs to a single field, so each
        dictionary is resolved by the option of the case among its keys.

    Returns
    -------
    `float` or `int`
        Value of the parameter for the case.
    """
    while isinstance(tree, dict):
        (option,) = [field for field in case if field in tree]
        tree = tree[option]
    return tree


def wave_dict(
    element_geometry: str,
    dimension: int,
    degree_eikonal: int,
    element_type: str,
    source_bc: str,
) -> dict:
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    degree_eikonal : `int`
        Finite element order for the Eikonal equation.
    element_type : `str`
        Finite element type. Options: "consistent" or "underintegrated".
    source_bc : `str`
        Dirichlet condition at the source, appended to the output folder.
        Options: "nearest_node" or "local_time".

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
        "analysis": "eikonal",  # Options: transient, modal or eikonal
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
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
    }

    # Define parameters for visualization
    str_ele = element_geometry + ("C" if element_type == "consistent" else "U")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/eikonal_test{dimension}d/eik_test{dimension}d"
        + f"{str_ele}_{source_bc}"
    }

    return dictionary


def eikonal_analysis(dictionary, edge_length, f_est, element_type, source_bc):
    """Run the the Eikonal analysis.

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    edge_length : `float`
        Mesh size in km.
    f_est : `float`
        Factor for the stabilizing term in Eikonal Eq.
    element_type : `string`
        Finite element type. Options: "consistent" or "underintegrated".
    source_bc : `str`
        Dirichlet condition at the source. Options: "nearest_node" or
        "local_time".

    Returns
    -------
    min_eik : `float`
        Minimum Eikonal value in miliseconds.
    """

    # ============ MESH FEATURES ============

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
    wave.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    wave.mesh_ops.preamble_mesh_operations(
        wave, ele_type_eik=element_type, f_est=f_est, source_bc_eik=source_bc
    )

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    wave.layer_ops.critical_boundary_points(wave)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "EIK_")

    # Extracting  minimum Eikonal
    min_eik = 1e3 * wave.layer_ops.eik_bnd[0][2]

    return min_eik


@mark.parametrize("source_bc", ["nearest_node", "local_time"])
@mark.parametrize("element_type", ["consistent", "underintegrated"])
@mark.parametrize("dimension", [2, param(3, marks=mark.slow)])
@mark.parametrize("element_geometry", ["T", "Q"])
def test_eikonal(element_geometry, dimension, element_type, source_bc):
    """Testing of eikonal for 2D and 3D case in Fig. 8 of Salas et al (2022).

    See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
    layers with non-reflecting boundary conditions in scalar wave equations.
    doi: https://doi.org/10.1016/j.apm.2022.09.014

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    element_type : `str`
        Finite element type. Options: "consistent" or "underintegrated".
    source_bc : `str`
        Dirichlet condition at the source. Options: "nearest_node" (zero at
        the nearest node) or "local_time" (local travel time within one cell
        of the source).

    Returns
    -------
    None

    Notes
    -----
    Theoretical value. eik_min is the first-arrival time on the absorbing
    boundaries (z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1; z = 0 is the
    free surface). The source is at (z, x) = (-0.5, 0.25) km ((z, x, y) =
    (-0.5, 0.25, 0.5) km in 3D). The velocity is 3.0 km/s for x < 0.5 and
    1.5 km/s for x >= 0.5. The first arrival on a boundary follows the straight
    ray to its nearest point, which gives 83.333 ms on x = 0 (0.25 km at
    3.0 km/s), 166.667 ms on z = -1 and on y = 0, 1, and 416.667 ms on x = 1.
    Hence eik_min = 83.333 ms, on x = 0 at (-0.5, 0.0) km ((-0.5, 0.0, 0.5) km
    in 3D).

    Selection of f_est. The mesh size (100 m in 2D, 150 m in 3D) and the
    Eikonal degree are fixed for each case, and f_est is calibrated separately
    for each source condition. The f_est used is the one with the smallest
    relative error abs(eik_min - 83.333) / 83.333 among the values for which
    the nonlinear solver converges with every supported Firedrake version
    (2025.4.1 and 2026.10.0). Values (ms) at the selected f_est, with the
    degree p in parentheses:

    Source        Case      2D                   3D
    nearest_node  T-cons    0.06  82.942 (2)     0.05  82.259 (2)
                  Q-cons    0.05  83.744 (2)     0.08  84.377 (2)
                  T-under   0.07  82.630 (2)     0.07  83.719 (2)
                  Q-under   0.03  84.245 (4)     0.06  83.476 (3)
    local_time    T-cons    0.12  83.347 (2)     0.04  85.533 (2)
                  Q-cons    0.12  83.357 (2)     0.02  85.710 (2)
                  T-under   0.08  83.100 (2)     0.07  85.720 (2)
                  Q-under   0.03  83.334 (4)     0.03  85.424 (3)

    For some f_est the nonlinear solver stalls, mostly with the underintegrated
    elements and the smaller f_est.
    Firedrake versions. The tables were computed with Firedrake 2025.4.1. With
    2026.10.0 all cases pass. For 'nearest_node' the minimum Eikonal at the
    selected f_est agrees within 0.01 ms, except T-3D consistent (82.243 vs
    82.259), T-3D underintegrated (83.710 vs 83.719) and Q-3D underintegrated
    (83.302 vs 83.476). For Q-3D underintegrated the critical point is
    (-0.532, 0, 0.532) km with 2026.10.0 and (-0.468, 0, 0.468) km with 2025.4.1:
    the source is snapped to four equidistant nodes, and the installations break
    the tie differently. For 'local_time' the values agree within 0.01 ms,
    except Q-3D underintegrated: f_est = 0.02 (85.061 ms, 2.07 %) does not
    converge with 2026.10.0, so the test uses f_est = 0.03, which converges
    with both versions (85.424 ms, 2.51 %).

    Complete documentation, with the Eikonal value at every f_est tried:
    tests/documentation_tests/test_eikonal__test_eikonal.md
    """

    SpyroEnsemble.print(
        "\n"
        + 60 * "="
        + f"\nTesting Eikonal with {element_geometry}-{element_type} "
        + f"elements for {dimension}D case and {source_bc} source\n"
        + 60 * "=",
    )

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km), Eikonal degree, tolerance and factor for the stabilizing
    # term in Eikonal equation of the case (see EIKONAL_CONFIG)
    case = (source_bc, dimension, element_type, element_geometry)
    edge_length = case_parameter(EIKONAL_CONFIG["edge_length"], case)
    p_eik = case_parameter(EIKONAL_CONFIG["p_eik"], case)
    atol = case_parameter(EIKONAL_CONFIG["atol"], case)
    f_est = case_parameter(EIKONAL_CONFIG["f_est"], case)

    # Get simulation parameters
    SpyroEnsemble.print(f"\nMesh Size: {1e3 * edge_length:.4f} m")
    SpyroEnsemble.print(f"Element Geometry: {element_geometry}")
    SpyroEnsemble.print(f"Element Type: {element_type}")
    SpyroEnsemble.print(f"Source Condition: {source_bc}")
    SpyroEnsemble.print(f"Eikonal Degree: {p_eik}")
    SpyroEnsemble.print(f"Eikonal Stabilizing Factor: {f_est:.2f}")

    try:
        # ============ MESH AND EIKONAL ============

        # Create dictionary with parameters for the model
        dictionary = wave_dict(
            element_geometry, dimension, p_eik, element_type, source_bc
        )

        # Creating mesh and performing eikonal analysis
        min_eik = round(
            eikonal_analysis(dictionary, edge_length, f_est, element_type, source_bc),
            3,
        )

        thr_val = 83.333  # in ms
        assert isclose(min_eik / thr_val, 1.0, atol=atol), (
            f"✗ Minimum Eikonal {dimension}D Element-{element_geometry}-"
            + f"{element_type} → Expected value {thr_val}, got {min_eik:.3f}"
        )
        SpyroEnsemble.print(
            f"✓ Minimum Eikonal {dimension}D Element-{element_geometry}-{element_type}"
            f" Verified: expected {thr_val}, got {min_eik:.3f}",
        )

    except ConvergenceError as e:
        fail(
            f"Checking Eikonal {dimension}D Element-{element_geometry}-"
            f"{element_type} raised an exception: {str(e)}"
        )
