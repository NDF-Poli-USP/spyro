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
    boundaries: z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1 (z = 0 is
    the free surface). The source is at (z, x) = (-0.5, 0.25) km in 2D and
    (z, x, y) = (-0.5, 0.25, 0.5) km in 3D. The velocity is 3.0 km/s for
    x < 0.5 and 1.5 km/s for x >= 0.5. The first arrival on a boundary
    follows the straight ray to its nearest point, since no other path is
    shorter and none reaches a faster medium:

    Boundary   Nearest point (z, x)   Path                          Time [ms]
    x = 0      (-0.5, 0.0)            0.25 km at 3.0 km/s              83.333
    z = -1     (-1.0, 0.25)           0.50 km at 3.0 km/s             166.667
    y = 0, 1   3D: (-0.5, 0.25, 0|1)  0.50 km at 3.0 km/s             166.667
    x = 1      (-0.5, 1.0)            0.25 km at 3.0 km/s
                                      + 0.50 km at 1.5 km/s           416.667

    eik_min = 0.25 km / 3.0 km/s = 83.333 ms, on x = 0 at (-0.5, 0.0) km
    (at (-0.5, 0.0, 0.5) km in 3D).

    eik_min = 83.333 ms (Theoretical value)

    Selection rule, applied separately for each source condition (source_bc).
    Successful configurations: the nonlinear solver converges at f_est - 0.01,
    f_est and f_est + 0.01 without retries, the three minimum Eikonal values
    stay within 10 % of each other and increase with f_est ('nearest_node')
    or change monotonically or by less than 0.1 % ('local_time', whose result
    barely depends on f_est), and the relative error
    abs(eik_min - 83.333) / 83.333 is below 3 %. For 'local_time' only meshes
    with N >= 5 are considered: with N <= 4 the source region (radius
    lmin = 1 / N >= 0.25 km) reaches the boundary x = 0, where the minimum
    Eikonal would be imposed instead of computed. N is the number of
    elements per direction (edge_length = 1 / N) and p the Eikonal degree.
    The winner (*) is the coarsest mesh, then the smallest relative error,
    among the configurations known to pass with every supported Firedrake
    version (see the end of these Notes).

    *******************************
    Source condition 'nearest_node'
    *******************************

    =========================
    2D - Consistent triangles
    =========================
     N  p  f_est  eik_min   error
     2  4  0.09    83.164   0.20 %*
     3  4  0.11    83.338   0.01 %
     4  3  0.02    83.120   0.26 %
     5  3  0.04    83.319   0.02 %
     5  4  0.08    83.304   0.03 %
     6  2  0.10    83.949   0.74 %
     6  3  0.16    83.544   0.25 %
     6  4  0.03    83.382   0.06 %
     7  3  0.14    83.056   0.33 %
     7  4  0.03    82.916   0.50 %
     8  1  0.10    83.206   0.15 %
     8  2  0.03    85.831   3.00 %
    10  2  0.06    82.942   0.47 %
    10  3  0.09    83.401   0.08 %
    10  4  0.03    84.047   0.86 %
    12  1  0.09    83.513   0.22 %
    12  2  0.03    85.637   2.77 %
    15  2  0.15    83.125   0.25 %
    15  3  0.08    83.276   0.07 %
    15  4  0.03    83.907   0.69 %
    20  1  0.06    83.329   0.01 %
    20  2  0.03    85.068   2.08 %
    20  3  0.03    85.791   2.95 %

    ==============================
    2D - Consistent quadrilaterals
    ==============================
     N  p  f_est  eik_min   error
     3  4  0.03    83.908   0.69 %*
     4  1  0.21    81.351   2.38 %
     5  4  0.09    83.335   0.00 %
     6  2  0.07    84.037   0.84 %
     6  3  0.15    83.352   0.02 %
     6  4  0.03    84.804   1.76 %
     7  2  0.81    81.744   1.91 %
     7  3  0.20    83.486   0.18 %
     7  4  0.02    85.409   2.49 %
     8  1  0.07    82.523   0.97 %
    10  1  0.56    83.320   0.02 %
    10  2  0.05    83.744   0.49 %
    10  3  0.09    83.433   0.12 %
    10  4  0.02    83.453   0.14 %
    12  1  0.07    83.797   0.56 %
    15  2  0.17    83.264   0.08 %
    15  3  0.09    83.285   0.06 %
    15  4  0.02    84.873   1.85 %
    20  1  0.05    83.257   0.09 %
    20  2  0.02    84.154   0.98 %
    20  4  0.02    85.634   2.76 %

    ==============================
    2D - Underintegrated triangles
    ==============================
     N  p  f_est  eik_min   error
     3  3  0.07    83.475   0.17 %*
     5  2  0.13    83.587   0.30 %
     5  3  0.11    83.111   0.27 %
     5  4  0.12    83.180   0.18 %
     6  2  0.11    84.113   0.94 %
     6  3  0.13    83.516   0.22 %
     7  2  0.24    83.423   0.11 %
     7  3  0.05    84.988   1.99 %
    10  2  0.08    84.272   1.13 %
    10  3  0.08    83.405   0.09 %
    15  2  0.11    83.379   0.06 %
    15  3  0.05    85.112   2.14 %

    ===================================
    2D - Underintegrated quadrilaterals
    ===================================
     N  p  f_est  eik_min   error
     3  1  0.01    83.789   0.55 %*
     5  4  0.11    83.289   0.05 %
     6  3  0.15    83.179   0.18 %
     6  4  0.04    85.582   2.70 %
     7  3  0.23    83.306   0.03 %
     8  1  0.19    83.243   0.11 %
    10  3  0.09    83.402   0.08 %
    10  4  0.04    85.593   2.71 %
    12  1  0.14    83.258   0.09 %
    15  2  0.20    83.315   0.02 %
    15  3  0.10    83.166   0.20 %
    20  1  0.13    85.045   2.05 %

    ==========================
    3D - Consistent tetrahedra
    ==========================
     N  p  f_est  eik_min   error
     2  3  0.14    84.673   1.61 %
     3  2  0.18    83.449   0.14 %
     3  3  0.18    82.671   0.79 %
     5  2  0.05    82.036   1.56 %
     6  2  0.04    83.499   0.20 %
     6  3  0.03    84.963   1.96 %
     7  2  0.05    82.070   1.52 %*
     7  3  0.05    82.159   1.41 %
     8  1  0.05    81.462   2.25 %

    =========================
    3D - Consistent hexahedra
    =========================
     N  p  f_est  eik_min   error
     2  3  0.10    85.386   2.46 %*
     3  3  0.16    82.970   0.44 %
     5  2  0.07    84.542   1.45 %
     5  3  0.05    83.563   0.28 %
     6  1  0.15    84.112   0.94 %
     7  2  0.08    84.377   1.25 %
     7  3  0.05    82.343   1.19 %

    ===============================
    3D - Underintegrated tetrahedra
    ===============================
     N  p  f_est  eik_min   error
     2  3  0.07    82.975   0.43 %
     4  1  0.20    82.203   1.36 %*
     5  2  0.07    82.403   1.12 %
     7  2  0.07    83.517   0.22 %

    ==============================
    3D - Underintegrated hexahedra
    ==============================
     N  p  f_est  eik_min   error
     4  1  0.21    84.007   0.81 %*
     5  3  0.06    85.008   2.01 %
     8  1  0.10    83.406   0.09 %

    *****************************
    Source condition 'local_time'
    *****************************

    =========================
    2D - Consistent triangles
    =========================
     N  p  f_est  eik_min   error
     5  4  0.05    80.904   2.92 %*
     6  3  0.05    82.474   1.03 %
     6  4  0.05    81.372   2.35 %
     7  1  0.07    81.712   1.94 %
     7  2  0.05    83.257   0.09 %
     7  3  0.05    82.614   0.86 %
     7  4  0.05    81.614   2.06 %
     8  2  0.02    84.070   0.88 %
     8  3  0.05    82.446   1.06 %
     8  4  0.05    82.801   0.64 %
    10  1  0.06    83.613   0.34 %
    10  2  0.03    83.477   0.17 %
    10  3  0.06    83.333   0.00 %
    10  4  0.05    83.169   0.20 %
    12  1  0.09    83.513   0.22 %
    12  2  0.02    83.504   0.21 %
    12  3  0.03    83.686   0.42 %
    12  4  0.02    83.525   0.23 %
    15  1  0.05    84.293   1.15 %
    15  2  0.04    83.869   0.64 %
    15  3  0.04    84.098   0.92 %
    15  4  0.03    83.647   0.38 %
    20  1  0.04    83.599   0.32 %
    20  2  0.02    84.027   0.83 %
    20  3  0.03    83.881   0.66 %
    20  4  0.03    83.838   0.61 %

    ==============================
    2D - Consistent quadrilaterals
    ==============================
     N  p  f_est  eik_min   error
     5  4  0.05    81.368   2.36 %*
     6  1  0.40    83.155   0.21 %
     6  3  0.05    82.256   1.29 %
     6  4  0.05    81.169   2.60 %
     7  1  0.05    85.184   2.22 %
     7  2  0.05    83.959   0.75 %
     7  3  0.05    82.591   0.89 %
     7  4  0.05    81.286   2.46 %
     8  2  0.05    83.090   0.29 %
     8  3  0.05    82.519   0.98 %
     8  4  0.05    82.509   0.99 %
    10  1  0.04    84.912   1.90 %
    10  2  0.03    83.293   0.05 %
    10  3  0.07    83.188   0.17 %
    10  4  0.05    83.327   0.01 %
    12  1  0.07    83.797   0.56 %
    12  2  0.03    83.445   0.13 %
    12  3  0.03    83.639   0.37 %
    12  4  0.02    83.399   0.08 %
    15  1  0.03    85.145   2.17 %
    15  2  0.02    83.902   0.68 %
    15  3  0.03    83.983   0.78 %
    15  4  0.02    83.495   0.19 %
    20  1  0.03    83.060   0.33 %
    20  2  0.02    83.746   0.50 %
    20  3  0.03    83.862   0.64 %
    20  4  0.02    83.629   0.36 %

    ==============================
    2D - Underintegrated triangles
    ==============================
     N  p  f_est  eik_min   error
     6  3  0.05    81.938   1.67 %*
     6  4  0.05    81.237   2.51 %
     7  2  0.08    82.155   1.41 %
     7  3  0.05    82.253   1.30 %
     7  4  0.05    81.769   1.88 %
     8  2  0.08    83.090   0.29 %
     8  3  0.08    81.788   1.85 %
     8  4  0.05    82.656   0.81 %
    10  2  0.08    83.100   0.28 %
    10  3  0.08    83.213   0.14 %
    10  4  0.05    83.274   0.07 %
    12  3  0.04    83.684   0.42 %
    12  4  0.04    83.666   0.40 %
    15  1  0.13    85.401   2.48 %
    15  2  0.10    84.724   1.67 %
    15  4  0.04    83.787   0.55 %
    20  2  0.10    85.039   2.05 %
    20  3  0.04    83.989   0.79 %
    20  4  0.05    84.127   0.95 %

    ===================================
    2D - Underintegrated quadrilaterals
    ===================================
     N  p  f_est  eik_min   error
     5  4  0.05    81.376   2.35 %*
     6  3  0.05    82.226   1.33 %
     6  4  0.05    81.168   2.60 %
     7  3  0.08    82.090   1.49 %
     7  4  0.06    81.018   2.78 %
     8  2  0.12    81.338   2.39 %
     8  3  0.08    82.061   1.53 %
     8  4  0.05    82.469   1.04 %
    10  1  0.19    83.282   0.06 %
    10  2  0.13    83.043   0.35 %
    10  3  0.08    83.159   0.21 %
    10  4  0.05    83.310   0.03 %
    12  1  0.14    83.258   0.09 %
    12  4  0.04    83.531   0.24 %
    15  1  0.13    84.019   0.82 %
    15  2  0.13    85.338   2.41 %
    15  4  0.05    83.808   0.57 %
    20  1  0.13    84.457   1.35 %
    20  2  0.12    85.051   2.06 %
    20  4  0.05    84.067   0.88 %

    ==========================
    3D - Consistent tetrahedra
    ==========================
     N  p  f_est  eik_min   error
     5  1  0.10    83.346   0.02 %*
     6  1  0.20    82.879   0.54 %
     6  3  0.05    84.374   1.25 %
     7  3  0.03    84.807   1.77 %
     8  1  0.33    83.458   0.15 %
     8  2  0.02    85.632   2.76 %
     8  3  0.03    84.418   1.30 %

    =========================
    3D - Consistent hexahedra
    =========================
     N  p  f_est  eik_min   error
     6  2  0.14    83.361   0.03 %*
     6  3  0.05    83.778   0.53 %
     7  3  0.02    85.018   2.02 %
     8  1  0.09    84.002   0.80 %
     8  3  0.02    84.505   1.41 %

    ===============================
    3D - Underintegrated tetrahedra
    ===============================
     N  p  f_est  eik_min   error
     5  3  0.05    81.183   2.58 %
     6  1  0.20    82.891   0.53 %*
     7  2  0.05    85.099   2.12 %
     7  3  0.04    84.376   1.25 %

    ==============================
    3D - Underintegrated hexahedra
    ==============================
     N  p  f_est  eik_min   error
     6  1  0.22    83.733   0.48 %
     6  3  0.06    83.384   0.06 %*
     7  3  0.03    85.424   2.51 %
     8  1  0.34    83.346   0.02 %
     8  3  0.04    85.223   2.27 %

    At the nodes shared by cells of different velocities, the velocity is the
    minimum of their values (``creating_velocity_profile``), so that it does not
    depend on the mesh partition.

    The tables were computed with Firedrake 2025.4.1. With Firedrake 2026.10.0
    the T-3D results can differ for two reasons. The velocity lives in the
    degree 3 KMV space of the wave, whose tetrahedron has another basis in
    2026.10.0 (see test_firedrake_elements.py), and the Eikonal evaluates it
    at its quadrature points: the Gauss points of the consistent elements
    and the quadrature of the Eikonal space for the underintegrated ones,
    which for p = 1 is at the vertices. In the cells at the interface, where
    the velocity is not constant, the values therefore differ, except for
    the underintegrated elements with p = 1, which only take nodal values.
    With an odd N, some tetrahedra also have their centroid exactly on the
    interface x = 0.5, where the cell velocity is decided by round-off in
    the condition x < 0.5 (50 of the 750 tetrahedra with N = 5).

    The T-3D winners are therefore taken among the configurations measured
    with 2026.10.0, or independent of the version (underintegrated, p = 1
    and an even N). Without this condition, the consistent 'nearest_node'
    winner would be N = 2, p = 3 (84.673 ms with 2025.4.1, whereas f_est =
    0.15 gives 91.428 ms with 2026.10.0, an error of 9.71 %), and the
    underintegrated winners would be N = 2 and N = 5 with p = 3, not
    measured with 2026.10.0. Every selected configuration (*) converges at
    its f_est without retries with 2026.10.0 and gives the same minimum
    Eikonal within 0.01 ms, except T-3D consistent (in ms, 2026.10.0 vs
    2025.4.1):

    Source condition   Case              2026.10.0   2025.4.1
    'nearest_node'     T-3D consistent      82.065     82.070
    'local_time'       T-3D consistent      82.633     83.346

    The 'nearest_node' value with 2026.10.0 is that of the same
    configuration in test_eikonal.py (N = 7, p = 2, f_est = 0.05). Both
    cases pass with both versions (atol = 2.5e-2 and 1.5e-2). The
    underintegrated T-3D winners give 82.203 ms ('nearest_node') and
    82.891 ms ('local_time') with both versions.
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
