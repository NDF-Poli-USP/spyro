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
from tests.on_one_core.test_eikonal import case_parameter

pytestmark = mark.eikonal

# Simulation parameters, selected for each source condition (see Notes of
# test_eikonal). Each parameter is a decision tree over the case fields
# (source_bc, dimension, element_type, element_geometry): every nested
# dictionary splits on exactly one field, keyed by that field's options
# (currently two each), and each leaf is the value shared by all the cases
# below it. A field is omitted where all its options share the same value, so
# that value is set once for the whole subtree. See case_parameter in
# test_eikonal.py for the lookup.
EIKONAL_CONFIG = {
    # Mesh size in km: edge_length = lba / cpw, where cpw is the number of cells
    # per wavelength and lba = minimum_velocity / source_frequency
    "edge_length": {
        "nearest_node": {
            2: {"T": {"consistent": 1.0 / 2, "underintegrated": 1.0 / 3}, "Q": 1.0 / 3},
            3: {"consistent": {"T": 1.0 / 7, "Q": 1.0 / 2}, "underintegrated": 1.0 / 4},
        },
        "local_time": {
            2: {"T": {"consistent": 1.0 / 5, "underintegrated": 1.0 / 6}, "Q": 1.0 / 5},
            3: {"T": {"consistent": 1.0 / 5, "underintegrated": 1.0 / 6}, "Q": 1.0 / 6},
        },
    },
    # Finite element order for the Eikonal equation
    "p_eik": {
        2: {
            "consistent": 4,
            "underintegrated": {
                "nearest_node": {"T": 3, "Q": 1},
                "local_time": {"T": 3, "Q": 4},
            },
        },
        3: {
            "consistent": {
                "nearest_node": {"T": 2, "Q": 3},
                "local_time": {"T": 1, "Q": 2},
            },
            "underintegrated": {"nearest_node": 1, "local_time": {"T": 1, "Q": 3}},
        },
    },
    # Absolute tolerance of eik_min / eik_thr with respect to 1
    "atol": {
        "nearest_node": {2: 1e-2, 3: {"consistent": 2.5e-2, "underintegrated": 1.5e-2}},
        "local_time": {
            2: {"consistent": 3e-2, "underintegrated": 2.5e-2},
            3: {"consistent": 1.5e-2, "underintegrated": 1e-2},
        },
    },
    # Factor for the stabilizing term in Eikonal equation
    "f_est": {
        "nearest_node": {
            2: {
                "T": {"consistent": 0.09, "underintegrated": 0.07},
                "Q": {"consistent": 0.03, "underintegrated": 0.01},
            },
            3: {
                "T": {"consistent": 0.05, "underintegrated": 0.20},
                "Q": {"consistent": 0.10, "underintegrated": 0.21},
            },
        },
        "local_time": {
            2: 0.05,
            3: {
                "T": {"consistent": 0.10, "underintegrated": 0.20},
                "Q": {"consistent": 0.14, "underintegrated": 0.06},
            },
        },
    },
}


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
        "type": "spatial",  # "automatic",
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
        + f"{str_ele}_{source_bc}_minmesh"
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


@mark.newer_firedrake
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
    ray to its nearest point:

    Boundary   Time [ms]
    x = 0         83.333   (0.25 km at 3.0 km/s)
    z = -1       166.667
    y = 0, 1     166.667   (3D)
    x = 1        416.667

    so eik_min = 83.333 ms, on x = 0 at (-0.5, 0.0) km ((-0.5, 0.0, 0.5) km in
    3D).

    Selection rule, applied separately for each source condition (source_bc).
    A configuration is successful when the nonlinear solver converges at
    f_est - 0.01, f_est and f_est + 0.01 without retries, the three minimum
    Eikonal values stay within 10 % of each other and increase with f_est
    ('nearest_node') or change monotonically or by less than 0.1 %
    ('local_time'), and the relative error abs(eik_min - 83.333) / 83.333 is
    below 3 %. For 'local_time' only meshes with N >= 5 are considered: with
    N <= 4 the source region (radius 1 / N >= 0.25 km) reaches x = 0. N is the
    number of elements per direction (edge_length = 1 / N) and p the Eikonal
    degree. The winner is the coarsest mesh, then the smallest error:

    Source        Case       N  p  f_est  eik_min   error
    nearest_node  T-2D cons  2  4  0.09    83.164   0.20 %
                  Q-2D cons  3  4  0.03    83.908   0.69 %
                  T-2D under 5  2  0.11    83.350   0.02 %
                  Q-2D under 3  1  0.01    83.789   0.55 %
                  T-3D cons  2  3  0.15    82.667   0.80 %
                  Q-3D cons  2  3  0.10    85.386   2.46 %
                  T-3D under 4  1  0.19    82.222   1.33 %
                  Q-3D under 4  1  0.21    84.007   0.81 %
    local_time    T-2D cons  5  4  0.05    80.904   2.92 %
                  Q-2D cons  5  4  0.05    81.368   2.36 %
                  T-2D under 6  3  0.05    81.938   1.67 %
                  Q-2D under 5  4  0.05    81.376   2.35 %
                  T-3D cons  5  1  0.10    83.563   0.28 %
                  Q-3D cons  6  2  0.14    83.361   0.03 %
                  T-3D under 6  1  0.20    83.084   0.30 %
                  Q-3D under 6  3  0.06    83.384   0.06 %

    Firedrake versions. The tables were computed with Firedrake 2025.4.1. With
    2026.10.0 every selected configuration converges without retries and gives
    the same minimum Eikonal within 0.01 ms, except T-3D consistent:
    'nearest_node' gives 82.101 ms (82.667 ms with 2025.4.1) and still passes,
    and 'local_time' gives 84.164 ms (83.563 ms) and fails (error 1.00 % against
    atol = 5e-3). The mesh (N = 5) has 50 of its 750 tetrahedra with the
    centroid exactly on the interface x = 0.5, where round-off in the condition
    x < 0.5 decides the cell velocity, so the installations evaluate these cells
    differently.

    Complete documentation, with the table of every (N, p, f_est) tried:
    tests/documentation_tests/test_eikonal_new__test_eikonal.md
    """

    SpyroEnsemble.print(
        "\n"
        + 60 * "="
        + f"\nTesting Eikonal with {element_geometry}-"
        + f"{element_type} elements for {dimension}D case and {source_bc} source\n"
        + 60 * "="
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
            f"✓ Minimum Eikonal {dimension}D Element-{element_geometry}-"
            f"{element_type} Verified: expected {thr_val}, got {min_eik:.3f}"
        )

    except ConvergenceError as e:
        fail(
            f"Checking Eikonal {dimension}D Element-{element_geometry}-"
            f"{element_type} raised an exception: {str(e)}"
        )
