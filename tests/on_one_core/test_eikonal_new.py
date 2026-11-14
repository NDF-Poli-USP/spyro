"""Unit tests for the Nonlinear Eikonal analysis for 2D and 3D cases.

The test compares the minimum Eikonal value obtained from the simulation with the
theoretical value for different mesh sizes, element geometries, and finite element types.
The results are expected to be within a specified tolerance of the theoretical value.
"""
from pytest import fail, mark
from firedrake import conditional, ConvergenceError
from numpy import isclose
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost


def wave_dict(element_geometry, dimension, degree_eikonal, element_type):
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
        + str_ele
    }

    return dictionary


def eikonal_analysis(dictionary, edge_length, f_est, element_type):
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
    wave.mesh_ops.preamble_mesh_operations(wave, ele_type_eik=element_type, f_est=f_est)

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
@mark.parametrize(
    "element_geometry, dimension, element_type",
    [
        ("T", 2, "consistent"),
        ("Q", 2, "consistent"),
        ("T", 2, "underintegrated"),
        ("Q", 2, "underintegrated"),
        ("T", 3, "consistent"),
        ("Q", 3, "consistent"),
        ("T", 3, "underintegrated"),
        ("Q", 3, "underintegrated"),
    ],
)
def test_eikonal(element_geometry, dimension, element_type):
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

    Returns
    -------
    None

    eik_min = 83.333 ms (Theoretical value)

    Successful configurations: the nonlinear solver converges at f_est - 0.01,
    f_est and f_est + 0.01 without retries, the three minimum Eikonal values
    increase with f_est and stay within 10 % of each other, and the relative
    error abs(eik_min - 83.333) / 83.333 is below 3 %. N is the number of
    elements per direction (edge_length = 1 / N) and p the Eikonal degree.
    The winner (*) is the coarsest mesh, then the smallest relative error.

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
     5  2  0.11    83.350   0.02 %*
     5  3  0.12    83.604   0.33 %
     5  4  0.12    83.207   0.15 %
     6  2  0.11    84.113   0.94 %
     6  3  0.13    83.516   0.22 %
     7  2  0.24    83.357   0.03 %
     7  3  0.05    84.738   1.69 %
    10  2  0.08    84.272   1.13 %
    10  3  0.08    83.405   0.09 %
    15  2  0.11    83.381   0.06 %
    15  3  0.05    85.108   2.13 %

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
     2  3  0.15    82.667   0.80 %*
     3  2  0.18    82.781   0.66 %
     3  3  0.19    83.896   0.68 %
     5  2  0.05    82.044   1.55 %
     6  2  0.04    84.603   1.52 %
     6  3  0.03    85.114   2.14 %
     7  2  0.05    82.259   1.29 %
     7  3  0.05    82.162   1.41 %
     8  1  0.05    82.516   0.98 %

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
     4  1  0.19    82.222   1.33 %*
     5  2  0.08    84.861   1.83 %

    ==============================
    3D - Underintegrated hexahedra
    ==============================
     N  p  f_est  eik_min   error
     4  1  0.21    84.007   0.81 %*
     5  3  0.06    85.008   2.01 %
     8  1  0.10    83.406   0.09 %
    """

    SpyroEnsemble.print(
        "\n"
        + 60 * "="
        + f"\nTesting Eikonal with {element_geometry}-"
        + f"{element_type} elements for {dimension}D case\n"
        + 60 * "="
    )

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw

    # Eikonal degree and factor for the stabilizing term in Eikonal equation
    if dimension == 2:
        atol = 1e-2
        if element_geometry == "T":
            edge_length = 1.0 / 2 if element_type == "consistent" else 1.0 / 5
            p_eik = 4 if element_type == "consistent" else 2
            f_est = 0.09 if element_type == "consistent" else 0.11
        else:
            edge_length = 1.0 / 3
            p_eik = 4 if element_type == "consistent" else 1
            f_est = 0.03 if element_type == "consistent" else 0.01

    if dimension == 3:
        edge_length = 1.0 / 2 if element_type == "consistent" else 1.0 / 4
        atol = 2.5e-2 if element_type == "consistent" else 1.5e-2
        if element_geometry == "T":
            p_eik = 3 if element_type == "consistent" else 1
            f_est = 0.15 if element_type == "consistent" else 0.19
        else:
            p_eik = 3 if element_type == "consistent" else 1
            f_est = 0.10 if element_type == "consistent" else 0.21

    # Get simulation parameters
    SpyroEnsemble.print(f"\nMesh Size: {1e3 * edge_length:.4f} m")
    SpyroEnsemble.print(f"Element Geometry: {element_geometry}")
    SpyroEnsemble.print(f"Element Type: {element_type}")
    SpyroEnsemble.print(f"Eikonal Degree: {p_eik}")
    SpyroEnsemble.print(f"Eikonal Stabilizing Factor: {f_est:.2f}")

    try:
        # ============ MESH AND EIKONAL ============

        # Create dictionary with parameters for the model
        dictionary = wave_dict(element_geometry, dimension, p_eik, element_type)

        # Creating mesh and performing eikonal analysis
        min_eik = round(
            eikonal_analysis(dictionary, edge_length, f_est, element_type), 3
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
