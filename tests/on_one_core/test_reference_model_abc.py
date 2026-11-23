"""Unit tests for the Reference Model implemented in spyro.abc.abc_layer.

These tests verify the consistency of the solver with HABCs and PML without any
damping for a model of reference with an extended pad to avoid reflections. The
tests are designed to ensure that the computed transiente responses and energies
are consistent with expected values. The tests cover both 2D and 3D cases.
"""

import itertools
from types import SimpleNamespace

import FIAT
import firedrake as fire
import numpy as np
import pytest
from finat.quadrature import make_quadrature
from firedrake import conditional, ConvergenceError
from numpy import all, isclose, sum
from spyro.abc.abc_layer import ABCLayer
from spyro.domains.quadrature import quadrature_rules
from spyro.domains.space import create_function_space
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.version_control import is_firedrake_new
from tests.on_one_core.test_eikonal import case_parameter

pytestmark = pytest.mark.habc

# Simulation parameters of test_infinite_model_abc (see its Notes). Each
# parameter is a decision tree over the case fields (dimension,
# element_geometry, timestep_method), looked up with case_parameter: every
# nested dictionary splits on exactly one field, keyed by that field's options,
# and each leaf is the value shared by all the cases below it.
REFERENCE_CONFIG = {
    # Mesh size in km
    "edge_length": {2: 0.25, 3: 0.5},
    # Factor for the stabilizing term in the Eikonal equation
    "f_est": {2: {"T": 0.03, "Q": 0.05}, 3: {"T": 0.66, "Q": 0.07}},
    # Index of the time step among the divisors of the final time that are
    # smaller than the estimated stable time step, in descending order. With
    # ARNOLDI the largest divisor is stable; with ANALYTICAL, the index skips
    # the divisors above the exact stable time step, so that both methods
    # select the same time step
    "max_divisor_tf": {
        "ANALYTICAL": {2: {"T": 3, "Q": 1}, 3: {"T": 4, "Q": 2}},
        "ARNOLDI": 1,
    },
    # Time step in s selected by check_timestep_abc, also the initial one
    "dt": {2: {"T": 0.004, "Q": 0.008}, 3: {"T": 0.01, "Q": 0.02}},
    # Size of the extension of the infinite model in km, with or without
    # Eikonal analysis
    "pad_length": {2: 3.0, 3: 2.5},
}

# Expected failure of the T-3D cases with newer Firedrake: the degree 3 KMV
# tetrahedron of firedrake-fiat 2026.10.0 has another basis than that of
# 2025.4.0 (see the Notes of test_infinite_model_abc). The 'ANALYTICAL' case
# becomes unstable and the 'ARNOLDI' case selects another time step, both
# raising AssertionError. strict=True reports the cases as soon as they pass.
KMV3_TETRAHEDRON_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the degree 3 KMV tetrahedron of newer Firedrake has another, "
    "non-symmetric basis, which makes the stable time step of T-3D about 11 "
    "times smaller",
)

# Stiffness diagonal of the degree 3 KMV tetrahedron on the UFC reference
# element, computed with its "KMV" quadrature, in the order of
# element.dual_basis(): 4 vertex, 12 edge, 12 face and 4 interior nodes (see
# test_kmv3_tetrahedron_reference_stiffness). The 2026.10.0 values were printed
# with four decimals.
KMV3_STIFFNESS_DIAGONAL = {
    "2025.4.1": (
        "0.08853607706 0.03741682612 0.03741682612 0.03741682612 "
        "0.3758670933 0.3758670933 0.3758670933 0.3758670933 0.3758670933 "
        "0.3758670933 0.4487776041 0.4396933483 0.4396933483 0.4487776041 "
        "0.4396933483 0.4487776041 2.660955652 2.660955652 2.660955652 "
        "1.689657416 1.689657416 1.667340012 1.689657416 1.689657416 "
        "1.667340012 1.689657416 1.689657416 1.667340012 5.980986394 "
        "5.980986394 5.980986394 6.202625231"
    ),
    "2026.10.0": (
        "1.9126 1.4815 2.0302 3.9271 14.4662 34.3064 34.3064 14.4662 13.3298 "
        "26.3461 54.6138 22.9601 15.1043 27.0778 8.9991 10.4341 14.3867 47.6151 "
        "14.3867 28.7872 103.6042 27.427 28.7872 103.6042 27.427 28.7872 "
        "103.6042 27.427 5.981 5.981 5.981 6.2026"
    ),
}
# Absolute tolerance of each recorded diagonal
KMV3_STIFFNESS_ATOL = {"2025.4.1": 1e-8, "2026.10.0": 1e-4}

# Stiffness diagonal of the spectral (Gauss-Lobatto-Legendre) elements of the Q
# cases on one unit cell, which is their reference cell: the degree 4
# quadrilateral (Q-2D) and the degree 3 quadrilateral x interval element of the
# extruded meshes (Q-3D), with spyro's quadrature. The dofs are grouped by their
# sorted distances to the faces of the cell, which a symmetry of the cell
# preserves (see test_spectral_element_reference_stiffness). Firedrake 2025.4.1.
SPECTRAL_STIFFNESS_DIAGONAL = {
    2: {
        (0.0, 0.0): 0.7,
        (0.0, 0.172673): 2.54074074074,
        (0.0, 0.5): 2.96296296296,
        (0.172673, 0.172673): 6.91646090535,
        (0.172673, 0.5): 7.09794238683,
        (0.5, 0.5): 6.74238683128,
    },
    3: {
        (0.0, 0.0, 0.0): 0.0902777777778,
        (0.0, 0.0, 0.276393): 0.358796296296,
        (0.0, 0.276393, 0.276393): 1.33101851852,
        (0.276393, 0.276393, 0.276393): 4.34027777778,
    },
}


def wave_dict(element_geometry, dimension, calc_eik, abc_type, dt_usu):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.
    abc_type : `str`
        Type of the absorbing boundary condition. Options: "hybrid" or "PML".
    dt_usu: `float`
        Time step of the simulation

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
        "analysis": "transient",  # Options: transient, modal or eikonal
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
        "delay_type": "multiples_of_minimum" if dimension == 2 else "time",
        "delay": 1.5 if dimension == 2 else 1.0 / 3.0,
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

    # Define parameters for the transient integration method.
    dictionary["time_axis"] = {
        "final_time": 2.0 if dimension == 2 else 1.5,  # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.0,  # the Ricker has an amplitude of 1.
        "output_frequency": 50,  # how frequently to output solution to pvds
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "abc_type": abc_type,  # Options: "hybrid" or "PML"
        "degree_eikonal": 2,  # FEM order for the Eikonal analysis
        "get_ref_model": True,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Eik" if calc_eik else "NoEik")
    output_folder = f"output/inf_test{dimension}d/inf_test{dimension}d" + str_ele
    dictionary["visualization"] = {  # Output folder
        "output_folder": output_folder,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": output_folder
        + f"/preamble/acoustic_energy_{abc_type}",
    }

    return dictionary


def wave_instance(
    element_geometry: str,
    dimension: int,
    abc_type: str,
    calc_eik: bool,
    timestep_method: str,
) -> AcousticWave:
    """Create an instance of the acoustic wave solver.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    abc_type : `str`
        Type of the absorbing boundary condition. Options: "hybrid" or "PML".
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.
    timestep_method : `str`
        Method of `Modal_Solver` for the stable time step, "ANALYTICAL" or
        "ARNOLDI". It selects the initial time step of REFERENCE_CONFIG.

    Returns
    -------
    wave : `AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """
    case = (dimension, element_geometry, timestep_method)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    edge_length = case_parameter(REFERENCE_CONFIG["edge_length"], case)

    # Factor for the stabilizing term in Eikonal equation
    f_est = case_parameter(REFERENCE_CONFIG["f_est"], case)

    # Timestep size (in seconds), the one selected by check_timestep_abc
    dt_usu = case_parameter(REFERENCE_CONFIG["dt"], case)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_geometry, dimension, calc_eik, abc_type, dt_usu)

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Get simulation parameters
    SpyroEnsemble.print(f"\nMesh Size: {1e3 * edge_length:.4f} m")
    SpyroEnsemble.print(f"Element Geometry: {element_geometry}")
    SpyroEnsemble.print(f"Eikonal Stabilizing Factor: {f_est:.2f}")
    SpyroEnsemble.print(f"Timestep Size: {1e3 * dt_usu:.3f} ms")

    # ============ MESH FEATURES ============

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
    wave.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations, with the Eikonal settings of test_eikonal
    wave.mesh_ops.preamble_mesh_operations(
        wave, ele_type_eik="consistent", f_est=f_est, source_bc_eik="nearest_node"
    )

    if calc_eik:
        # ============ EIKONAL ANALYSIS ============

        # Finding critical points
        wave.layer_ops.critical_boundary_points(wave)

    return wave


@pytest.mark.parametrize("timestep_method", ["ANALYTICAL", "ARNOLDI"])
@pytest.mark.parametrize("calc_eik", [True, False])
@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.slow)])
@pytest.mark.parametrize("element_geometry", ["T", "Q"])
def test_infinite_model_abc(
    request: pytest.FixtureRequest,
    element_geometry: str,
    dimension: int,
    calc_eik: bool,
    timestep_method: str,
) -> None:
    """Check the reference model of the ABCs for 2D and 3D cases.

    The reference model is the one of Fig. 8 of Salas et al (2022): the
    original domain extended at the bottom and on the lateral boundaries, far
    enough that no reflection reaches the receivers before the final time. It
    is built and solved with the hybrid and with the PML scheme, which must
    give the same receiver signals and energies, since the extension has no
    damping.

    See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
    layers with non-reflecting boundary conditions in scalar wave equations.
    doi: https://doi.org/10.1016/j.apm.2022.09.014

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Fixture that marks the T-3D cases as expected failures with newer
        Firedrake (KMV3_TETRAHEDRON_XFAIL).
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.
    timestep_method : `str`
        Method of `Modal_Solver` for the stable time step: "ANALYTICAL" (largest
        ratio of the stiffness and mass diagonals) or "ARNOLDI" (exact largest
        eigenvalue).

    Returns
    -------
    None

    Notes
    -----
    Stages. Each case runs four stages, and the test checks each of them:

    1. Eikonal analysis (``calc_eik``): the minimum Eikonal on the absorbing
       boundaries, with the settings of ``test_eikonal`` (degree 2, consistent
       element, 'nearest_node' source condition). Its theoretical value is
       83.333 ms (see the Notes of ``test_eikonal``).
    2. Time step (``check_timestep_abc``): the stable time step
       ``2 / sqrt(lambda_max)`` of the forms of the wave physics, and then the
       divisor of the final time selected by ``max_divisor_tf``. The selected
       time step must be the one of REFERENCE_CONFIG["dt"].
    3. Extension (``layer_infinite_model``): its size must be the one of
       REFERENCE_CONFIG["pad_length"].
    4. Forward solve with both schemes: the integral and peak errors between
       their receiver signals must be zero, and so must the difference of their
       final energies.

    Selection of f_est. As in ``test_eikonal``, the marked value (*) is the
    f_est with the smallest relative error abs(eik_min - 83.333) / 83.333
    among the values for which the nonlinear solver converges, and is the one
    used by the test. --/--: the nonlinear solver stalls at this f_est (it
    then retries with f_est + 0.01). In 2D Q, f_est = 0.02 converges to 27.174 ms,
    a spurious solution far below the theoretical value. These meshes are coarse
    (the 0.25 km from the source to x = 0 are one element in 2D and half an element
    in 3D), so the errors are larger than in ``test_eikonal``; for T-3D the best
    value is 35 % below the theoretical one. Values with Firedrake 2025.4.1.

    ==============================
    Eikonal for 2D model Δx = 250m
    ==============================
    eik_min = 83.333 ms

    f_est  T-ele   Q-ele
     0.01  --/--   --/--
     0.02  --/--  27.174
     0.03 84.586*  --/--
     0.04 87.945   --/--
     0.05 90.748  93.368*
     0.06 93.078  97.456
     0.07 95.040 100.718

    ==============================
    Eikonal for 3D model Δx = 500m
    ==============================
    eik_min = 83.333 ms

    f_est  T-ele f_est   Q-ele
     0.62 53.873  0.05  48.938
     0.64 53.939  0.06  63.319
     0.65 53.956  0.07  78.381*
     0.66 53.962* 0.08  93.750
     0.67 53.958  0.09 109.148
     0.68 53.946  0.10 124.374

    Size of the extension. A wave that first reaches the original boundary at
    ``t_bnd`` returns there, after a reflection at the boundary of the
    extension, at ``t_bnd + 2 * pad / c_bnd_max``. The extension is therefore
    ``c_bnd_max * (final_time - t_bnd) / 2``, rounded up to a multiple of
    ``lmin``, with ``t_bnd`` the minimum Eikonal or, without Eikonal analysis,
    the distance from the source to the nearest boundary (0.25 km, on x = 0)
    over ``c_bnd_max`` (3.0 km/s). Both give 3.0 km in 2D and 2.5 km in 3D,
    for every f_est of the tables.

    Time step. The final time is a multiple of the time step, so the time step
    is a divisor of the final time, taken below the estimated stable time step
    (``fraction = 1``); ``max_divisor_tf`` selects the divisor, largest first.
    'ANALYTICAL' takes the largest eigenvalue as the largest ratio of the
    stiffness and mass diagonals. Each ratio is the Rayleigh quotient of a unit
    vector, so it never exceeds the largest eigenvalue, and the estimated time
    step is never below the stable one: here it is 14 to 50 % above it.
    'ARNOLDI' computes the largest eigenvalue, and the largest divisor below
    its estimate is stable, so it uses the first divisor. 'ANALYTICAL' uses the
    first divisor that is not above the exact stable time step, so both methods
    select the same time step: the third divisor for T-2D, the first for Q-2D,
    the fourth for T-3D and the second for Q-3D.
    The time steps (in ms) were computed with the forms of the wave physics,
    on the original mesh (where ``check_timestep_abc`` computes them) and on
    the mesh of the infinite model (exact value only); the last two columns
    are the selected time steps and their ratio to the exact stable time step
    of the infinite model. The last table gives the time to estimate the stable
    time step as ``check_timestep_abc`` does (original mesh, ``fraction = 1``),
    in s: the first call with empty compilation caches, each method in its own
    process, and the median of three further calls ("warm"). The first call is
    mostly the compilation of the forms, the same for both methods. Afterwards,
    'ARNOLDI' takes 2 to 3 times as long as 'ANALYTICAL', but both take less
    than 0.02 s, against 9 to 46 s for the forward solves of a case.
    ``check_timestep_abc`` prints the method, the estimate and the time of its
    own call, the divisors from the largest down to the selected one, and the
    ratio of the selected time step to the estimate. Its call is the first of
    the process, which loads the compiled kernels: 0.07 to 0.09 s for Q-2D
    with filled caches.

    ===================================
    Time step for 2D model Δx = 250m
    ===================================
    final_time = 2.0 s

    Case  ANALYTICAL  Exact (orig.)  Exact (inf.)  Divisors <= ANALYTICAL
    T-2D     6.921        4.614          4.603     6.25  5.00  4.00  3.20
    Q-2D     9.960        8.725          8.704     8.00  6.25  5.00  4.00

    Case  'ANALYTICAL'                'ARNOLDI' (1st divisor)
    T-2D  3rd divisor  4.00  0.87           4.00  0.87
    Q-2D  1st divisor  8.00  0.92           8.00  0.92

    Case  ndof  'ANALYTICAL' first  warm  'ARNOLDI' first  warm
    T-2D   385              0.763  0.004            0.768  0.010
    Q-2D   289              1.101  0.005            1.068  0.013

    ===================================
    Time step for 3D model Δx = 500m
    ===================================
    final_time = 1.5 s

    Case  ANALYTICAL  Exact (orig.)  Exact (inf.)  Divisors <= ANALYTICAL
    T-3D    15.945       11.152         11.136    15.0  12.5  12.0  10.0
    Q-3D    26.688       22.640         22.339    25.0  20.0  15.0  12.5

    Case  'ANALYTICAL'                'ARNOLDI' (1st divisor)
    T-3D  4th divisor 10.00  0.90          10.00  0.90
    Q-3D  2nd divisor 20.00  0.90          20.00  0.90

    Case  ndof  'ANALYTICAL' first  warm  'ARNOLDI' first  warm
    T-3D   775              1.024  0.007            0.994  0.018
    Q-3D   343              2.948  0.007            3.077  0.014

    With 'ANALYTICAL', every divisor above the exact stable time step would be
    unstable: 6.25 and 5.00 ms for T-2D, and 15.0, 12.5 and 12.0 ms for T-3D.
    The deprecated forms of ``estimate_timestep``, with the
    velocity in the stiffness, gave the same estimates except for Q-3D (26.752
    ms), with the same divisors.

    Time. Here, with Firedrake 2025.4.1 on one core, a 2D case takes 9 to 31 s
    and a 3D case 13 to 46 s; the 3D cases are slow tests.

    ====================================
    Firedrake 2026.10.0 against 2025.4.1
    ====================================
    2D and Q-3D. The estimated time steps, the divisors and the selected time
    steps are the same, and the cases pass. The minimum Eikonal agrees within
    0.05 ms: T-2D 84.544 ms (84.586), Q-2D 93.368 ms, Q-3D 78.383 ms (78.381).

    T-3D. On the same mesh (48 tetrahedra), the stable time step is about 11
    times smaller, so the largest eigenvalue is about 127 times larger, and the
    four T-3D cases fail:

    Version   ANALYTICAL  Exact (orig.)  Divisors <= ANALYTICAL  'ANA'  'ARN'
    2025.4.1    15.945       11.152      15.0  12.5  12.0  10.0  10.0   10.0
    2026.10.0    1.885        0.989      1.50  1.25  1.20  1.00   1.0    0.8

    With 'ANALYTICAL', the fourth divisor (1.0 ms) is above the exact stable
    time step (0.989 ms), and the forward solve stops with "Numerical
    instability". 'ARNOLDI' selects 0.8 ms and runs to the final time, but the
    check of the selected time step (10 ms) fails. The minimum Eikonal is
    53.454 ms (53.962). The cause is the degree 3 KMV tetrahedron of FIAT.
    With both versions the T-3D space has 775 dofs and the same diagonal mass
    matrix (1.908e-05 to 1.190e-03), but the stiffness diagonal differs: at
    the node (0, 0.157, 0.157) it is 47.2 with 2026.10.0 and 0.508 with
    2025.4.1 (whose largest entry is 3.05). Over all 775 dofs, paired by
    their coordinates (the numbering differs between the versions), the mass
    diagonal is identical, and the stiffness diagonal is unchanged only at the
    192 dofs inside the cells (4 per cell); at the 583 vertex, edge and face
    dofs it is 6 to 132 times larger (median 12 to 47 per kind of dof), both
    on the boundary of the domain and inside it, and its sum grows from 1257.4
    to 17267.7. On the reference element, both
    versions have the same 32 nodes, the same quadrature and the same lumped
    mass, and the basis is nodal, but the stiffness diagonal sums to 865.8
    with 2026.10.0 against 52.4 with 2025.4.1; only the four interior nodes
    agree. The 2026.10.0 basis is also not symmetric: vertices 1, 2 and 3,
    which a permutation of the axes exchanges, give 1.48, 2.03 and 3.93,
    against 0.0374 for each with 2025.4.1. The basis functions therefore
    differ between the nodes, which points to an error in the element of
    2026.10.0. The T-3D parameters hold for 2025.4.1.

    Elements involved. Of the elements of this test, only the degree 3 KMV
    tetrahedron (T-3D) changed: the degree 4 KMV triangle (T-2D) and the
    spectral (Gauss-Lobatto-Legendre) quadrilateral and hexahedron of degrees
    4 and 3 (Q-2D, Q-3D) give the same time steps with both versions
    (test_spectral_element_reference_stiffness shows that the latter two are
    also identical on the reference cell). Other
    degrees of the KMV tetrahedron were not checked. Every spyro simulation
    with lumped degree 3 tetrahedra is affected with 2026.10.0, not only this
    test. With newer Firedrake (``is_firedrake_new()``, version 2026.4 or later;
    only 2026.10.0 was checked), the four T-3D cases are therefore expected
    failures (KMV3_TETRAHEDRON_XFAIL, strict): they still run, and are reported
    as soon as they pass, for example once the element is corrected upstream.
    With the mark, 2026.10.0 gives 27 passed and 4 xfailed (210 s): 'ANALYTICAL'
    stops at the instability after 0.1 s of simulation, and 'ARNOLDI' at the
    check of the selected time step. test_kmv3_tetrahedron_reference_stiffness
    reproduces the difference of the element itself, with FIAT and FInAT only.

    The time printed by ``check_timestep_abc`` with 2026.10.0 is 2.6 to 10.6 s
    for the first estimate of a geometry and dimension (compilation), and then
    0.05 to 0.12 s for 'ANALYTICAL' and 0.07 to 0.15 s for 'ARNOLDI'.
    """
    if element_geometry == "T" and dimension == 3 and is_firedrake_new():
        request.applymarker(KMV3_TETRAHEDRON_XFAIL)

    act_eik = "Activated" if calc_eik else "Deactivated"
    case = (dimension, element_geometry, timestep_method)
    max_divisor_tf = case_parameter(REFERENCE_CONFIG["max_divisor_tf"], case)
    expected_dt = case_parameter(REFERENCE_CONFIG["dt"], case)
    expected_pad = case_parameter(REFERENCE_CONFIG["pad_length"], case)

    # ============ REFERENCE MODEL ============

    # Initialize variables to store hybrid and PML results
    hybrid_signal = None
    pml_signal = None
    hybrid_energy = None
    pml_energy = None

    try:
        for abc_type in ["hybrid", "PML"]:
            # Reference to resource usage
            tRef = comp_cost("tini")

            # Create an instance of the acoustic wave solver
            wave = wave_instance(
                element_geometry, dimension, abc_type, calc_eik, timestep_method
            )

            SpyroEnsemble.print(f"\nAbsorbing Boundary Condition: {abc_type}")
            SpyroEnsemble.print(f"Index of the Divisor of Final Time: {max_divisor_tf}")

            SpyroEnsemble.print(
                "\n"
                + 60 * "="
                + f"\nTesting Reference Model with {element_geometry} "
                + f"elements for ABCs\nand {dimension}D case. Eikonal analysis: {act_eik}\n"
                + 60 * "="
            )

            # Computing reference get_reference_signal
            wave.layer_ops.infinite_model(
                wave,
                check_dt=True,
                max_divisor_tf=max_divisor_tf,
                method=timestep_method,
            )

            # Selected time step and size of the extension
            assert isclose(wave.dt, expected_dt, rtol=1e-12), (
                f"✗ Time step {1e3 * wave.dt} ms selected with {timestep_method}, "
                f"expected {1e3 * expected_dt} ms"
            )
            assert isclose(wave.layer_ops.abc_pad_length, expected_pad, rtol=1e-12), (
                f"✗ Extension of {wave.layer_ops.abc_pad_length} km, expected "
                f"{expected_pad} km"
            )

            (
                receivers_reference,
                receivers_ref_fft,
            ) = wave.layer_ops.get_reference_signal()

            # Estimating computational resource usage
            comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "preamble/INF_")

            if abc_type == "hybrid":
                hybrid_signal = receivers_reference
                hybrid_energy = wave.simulation_logger.get("acoustic_energy")
            else:
                pml_signal = receivers_reference
                pml_energy = wave.simulation_logger.get("acoustic_energy")

        # Checking both signals
        assert hybrid_signal is not None, "Hybrid signal not found"
        assert pml_signal is not None, "PML signal not found"

        dt = wave.dt
        error_measures = wave.layer_ops.calculate_error_measures(
            pml_signal,
            hybrid_signal,
            dt,
            wave.number_of_receivers,
            final_energy=pml_energy,
            final_energy_reference=hybrid_energy,
            save_in_case_folder=False,
        )

        (
            errIt,
            errPk,
            pkMax,
            max_errIt,
            max_errPK,
            final_ener,
            dsspt_ener,
        ) = error_measures

        assert sum(errIt) == 0.0 and max_errIt == 0.0, (
            "✗ Integral Error check for 'hybrid' and 'PML' solvers in Reference Model "
            f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        )
        SpyroEnsemble.print("✓ Integral Error Verified for 'hybrid' and 'PML' solvers")
        assert sum(errPk) == 0.0 and max_errPK == 0.0 and all(pkMax) > 0.0, (
            "✗ Peak Error check for 'hybrid' and 'PML' solvers in Reference Model "
            f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        )
        SpyroEnsemble.print("✓ Peak Error Verified for 'hybrid' and 'PML' solvers")
        assert final_ener > 0.0 and dsspt_ener == 0.0, (
            "✗ Final Energy check for 'hybrid' and 'PML' solvers in Reference Model "
            f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        )
        SpyroEnsemble.print("✓ Final Energy Verified for 'hybrid' and 'PML' solvers")

    except ConvergenceError as e:
        pytest.fail(
            f"Checking Reference Model with {element_geometry} elements for "
            f"{dimension}D and Eikonal {act_eik} case raised an exception: {str(e)}",
        )


def abc_layer(dimension: int, dt: float | None = None) -> ABCLayer:
    """Create a rectangular ABC layer for the unit square or the unit cube.

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    dt : `float`, optional
        Initial time step in s. Default is `None`.

    Returns
    -------
    `ABCLayer`
        Layer whose original domain is 1 km long in every direction.
    """
    return ABCLayer(
        (1.0,) * dimension,
        dt=dt,
        dimension=dimension,
        output_folder="output/inf_model_unit",
    )


# Velocity, final time and mesh size for the extension of the infinite model.
# lmin is chosen so that no required extension below is a multiple of it.
C_BND_MAX, FINAL_TIME, LMIN = 3.0, 2.0, 0.04


@pytest.mark.parametrize(
    "dimension, sources",
    [
        (2, [(-0.5, 0.25)]),  # source of test_infinite_model_abc, nearest x = 0
        (2, [(-0.9, 0.5)]),  # nearest the bottom
        (2, [(-0.2, 0.6), (-0.5, 0.25)]),  # two sources, the second is nearest
        (3, [(-0.5, 0.25, 0.5)]),  # source of test_infinite_model_abc
        (3, [(-0.5, 0.5, 0.1)]),  # nearest y = 0
        (3, [(-0.8, 0.5, 0.5), (-0.5, 0.5, 0.95)]),  # the second is nearest y = 1
    ],
)
def test_layer_infinite_model_without_eikonal(dimension: int, sources: list) -> None:
    """Check that no reflection returns before the final time without Eikonal.

    Without Eikonal analysis, the earliest time at which the wave reaches the
    absorbing boundaries is taken as the distance from the nearest source to
    the nearest of them (the bottom z = -1 and the lateral boundaries; z = 0
    is the free surface) over ``c_bnd_max``. A wave reflected by the boundary
    of the extension returns to the original boundary ``2 * pad / c_bnd_max``
    later, which must not be before the final time, and the extension must not
    be longer than needed by more than ``lmin``.

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    sources : `list`
        Source locations (z, x) or (z, x, y) in km.

    Returns
    -------
    None

    Notes
    -----
    Before the correction of ``layer_infinite_model``, the distance to the
    bottom was ``abs(z - length_z)`` instead of ``abs(z + length_z)``, the whole
    distance was subtracted instead of half of it, and two sources raised
    ``ValueError``. The first two cases then gave reflections back at 1.917 s
    and 1.700 s, before the final time of 2.0 s.
    """
    # Distance from the nearest source to the nearest absorbing boundary
    dist = min(
        min([1.0 + src[0]] + [min(coord, 1.0 - coord) for coord in src[1:]])
        for src in sources
    )
    t_bnd = dist / C_BND_MAX
    needed = C_BND_MAX * (FINAL_TIME - t_bnd) / 2.0

    pad = abc_layer(dimension).layer_infinite_model(
        LMIN, C_BND_MAX, FINAL_TIME, source_locations=sources
    )

    assert t_bnd + 2.0 * pad / C_BND_MAX >= FINAL_TIME - 1e-12
    assert needed - 1e-12 <= pad < needed + LMIN
    assert isclose(pad / LMIN, round(pad / LMIN))


@pytest.mark.parametrize("eikmin", [0.084586, 0.053962])
def test_layer_infinite_model_with_eikonal(eikmin: float) -> None:
    """Check that no reflection returns before the final time with Eikonal.

    With Eikonal analysis, the earliest time at which the wave reaches the
    absorbing boundaries is the minimum Eikonal at the critical boundary
    points, and the source locations are not used.

    Parameters
    ----------
    eikmin : `float`
        Minimum Eikonal in s, those of T-2D and T-3D in test_infinite_model_abc.

    Returns
    -------
    None
    """
    layer = abc_layer(2)
    # Structure of eik_bnd: [pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]
    layer.eik_bnd = [[(-0.5, 0.0), C_BND_MAX, eikmin]]
    needed = C_BND_MAX * (FINAL_TIME - eikmin) / 2.0

    pad = layer.layer_infinite_model(
        LMIN, C_BND_MAX, FINAL_TIME, source_locations=[(-0.9, 0.5)]
    )

    assert eikmin + 2.0 * pad / C_BND_MAX >= FINAL_TIME - 1e-12
    assert needed - 1e-12 <= pad < needed + LMIN


@pytest.mark.parametrize(
    "estimate, max_divisor_tf, set_max_dt, expected_dt",
    [
        (6.9205e-3, 1, True, 6.25e-3),  # T-2D, ANALYTICAL, largest divisor
        (6.9205e-3, 3, True, 4.0e-3),  # T-2D, ANALYTICAL, as in the test
        (4.6135e-3, 1, True, 4.0e-3),  # T-2D, ARNOLDI
        (9.9602e-3, 3, True, 5.0e-3),  # Q-2D, ANALYTICAL, third divisor
        (9.9602e-3, 1, True, 8.0e-3),  # Q-2D, ANALYTICAL, as in the test
        (8.7247e-3, 1, True, 8.0e-3),  # Q-2D, ARNOLDI
        (6.9205e-3, 1, False, 4.0e-3),  # current time step is smaller
    ],
)
def test_check_timestep_abc_selects_divisor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    estimate: float,
    max_divisor_tf: int,
    set_max_dt: bool,
    expected_dt: float,
) -> None:
    """Check the time step selected among the divisors of the final time.

    The stable time step is replaced by a fixed estimate, those measured in
    test_infinite_model_abc, so the test checks the selection alone: the
    divisors of the final time below the estimate, largest first, the forms
    of the wave physics and the full estimate (``fraction = 1``).

    Parameters
    ----------
    monkeypatch : `pytest.MonkeyPatch`
        Fixture that replaces `Modal_Solver` in `spyro.abc.abc_layer`.
    capsys : `pytest.CaptureFixture`
        Fixture that captures the printed information.
    estimate : `float`
        Estimated stable time step in s.
    max_divisor_tf : `int`
        Index of the selected divisor, largest first.
    set_max_dt : `bool`
        If `True`, the selected divisor is set; otherwise, the smaller of it and
        the current time step (4 ms).
    expected_dt : `float`
        Expected time step in s.

    Returns
    -------
    None
    """
    calls = []
    forms = ("stiffness form", "mass form")

    class FixedEstimate:
        """Stand-in for `Modal_Solver` that returns ``estimate``."""

        def __init__(self, dimension: int, method: str, calc_max_dt: bool) -> None:
            """Record the solver options.

            Parameters
            ----------
            dimension : `int`
                Dimension of the problem.
            method : `str`
                Method for the largest eigenvalue.
            calc_max_dt : `bool`
                If `True`, the largest eigenvalue is sought.

            Returns
            -------
            None
            """
            calls.append({"method": method, "calc_max_dt": calc_max_dt})

        def estimate_timestep(self, *args, fraction: float, forms, **kwargs) -> float:
            """Record the fraction and the forms, and return ``estimate``.

            Parameters
            ----------
            *args : `tuple`
                Positional arguments of `Modal_Solver.estimate_timestep`.
            fraction : `float`
                Fraction of the estimated time step.
            forms : `tuple`
                Stiffness and mass forms.
            **kwargs : `dict`
                Other keyword arguments of `Modal_Solver.estimate_timestep`.

            Returns
            -------
            `float`
                The fixed estimate.
            """
            calls[-1].update(fraction=fraction, forms=forms)
            return estimate

    monkeypatch.setattr("spyro.abc.abc_layer.Modal_Solver", FixedEstimate)
    wave = SimpleNamespace(
        dt=4.0e-3, final_time=FINAL_TIME, c=None, function_space=None
    )
    wave.quadrature_rule = None
    wave.get_dt = lambda: wave.dt
    wave.set_dt = lambda dt: setattr(wave, "dt", dt)
    wave.modal_weak_forms = lambda: forms
    layer = abc_layer(2, dt=wave.dt)

    layer.check_timestep_abc(
        wave, max_divisor_tf=max_divisor_tf, set_max_dt=set_max_dt, method="ARNOLDI"
    )

    assert isclose(wave.dt, expected_dt, rtol=1e-12)
    assert isclose(layer.freq_Nyquist, 1.0 / (2.0 * expected_dt), rtol=1e-12)
    assert calls == [
        {"method": "ARNOLDI", "calc_max_dt": True, "fraction": 1.0, "forms": forms}
    ]

    # Printed information: method, estimate, divisors down to the selected one
    out = capsys.readouterr().out
    assert "Method for the Maximum Timestep Size: ARNOLDI" in out
    assert f"Estimated Maximum Timestep Size: {1e3 * estimate:.3f} ms" in out
    selected = 1e3 * expected_dt if set_max_dt else 6.25
    divisors_line = next(
        line for line in out.splitlines() if "Largest Divisors" in line
    )
    assert divisors_line.endswith(f"{selected:.3f}")
    assert len(divisors_line.split(": ")[1].split(", ")) == max_divisor_tf


def test_kmv3_tetrahedron_reference_stiffness(request: pytest.FixtureRequest) -> None:
    """Check the basis of the degree 3 KMV tetrahedron on the reference element.

    The T-3D cases of test_infinite_model_abc use this element. With Firedrake
    2026.10.0 their stable time step is about 11 times smaller than with
    2025.4.1 (see the Notes of test_infinite_model_abc). This test isolates the
    cause with FIAT and FInAT only: it builds the element and its "KMV"
    quadrature on the UFC reference tetrahedron, and the mass and stiffness
    matrices with that quadrature. Run it with ``pytest -s`` to print the
    stiffness diagonal node by node, next to the values recorded with 2025.4.1
    and 2026.10.0 (KMV3_STIFFNESS_DIAGONAL), so that each version shows how it
    differs from the other.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Fixture that marks the test as an expected failure with newer Firedrake
        (KMV3_TETRAHEDRON_XFAIL).

    Returns
    -------
    None

    Notes
    -----
    Both versions agree on the 32 nodes (4 at the vertices, 2 on each edge, 3
    on each face, 4 interior), on the quadrature, whose points are the nodes,
    on the diagonal mass matrix, whose entries are the quadrature weights, and
    on the nodal basis (identity at the nodes). The test checks these first.
    The stiffness diagonal differs: it sums to 52.39 with 2025.4.1 and to
    865.75 with 2026.10.0, and only the four interior nodes keep their values.
    A permutation of the axes maps the reference tetrahedron onto itself and
    permutes vertices 1, 2 and 3, so their stiffness entries must be equal:
    0.0374 each with 2025.4.1, but 1.4815, 2.0302 and 3.9271 with 2026.10.0.
    The test also tabulates the basis at 35 points of a lattice inside the
    cell, between the nodes, and checks two properties that hold for any
    correct degree 3 KMV tetrahedron, without recorded values: the basis
    interpolates every polynomial of degree 3 or less exactly (both versions
    pass), and a swap P of two axes, which maps the cell and the nodes onto
    themselves, satisfies phi_i(P x) = phi_P(i)(x) (2026.10.0 fails, see
    below). Then it requires the symmetry of the stiffness diagonal of
    vertices 1, 2 and 3 and the 2025.4.1 values, so it fails with 2026.10.0,
    first at the symmetry of the basis, and with ``strict=True`` it is
    reported as soon as it passes there.

    Results. With 2025.4.1 the test passes (6 s); the current diagonal matches
    the 2025.4.1 values to 3.3e-10 and differs from the 2026.10.0 values at the
    28 vertex, edge and face nodes, by up to 101.9. With 2026.10.0 it is an
    expected failure (xfailed, 5.6 s); the current diagonal matches the
    2026.10.0 values to 4.8e-05, within their four-decimal rounding, and
    differs from the 2025.4.1 values at the same 28 nodes, by up to 101.9. The
    checks shared by both versions pass with both: the cubics are reproduced
    to 1.1e-15 (2025.4.1) and 1.0e-15 (2026.10.0). The symmetry of the basis
    holds to 1.0e-15 with 2025.4.1; with 2026.10.0 it fails by 2.48 for each
    swap (vertex 0.31 to 0.50, edge 1.34 to 1.44, face 2.48, interior 5.6e-16),
    where the test stops as an expected failure (8.7 s for both element tests
    here, 18.1 s there).

    Degree. The element is of degree 3: it contains every polynomial of degree
    3 or less. To have positive quadrature weights at its nodes (mass
    lumping), it adds bubble functions up to degree 5, which vanish on the
    boundary of the cell, so FIAT reports ``degree() == 5``. Its 32 nodes are
    the 20 of the cubic Lagrange element (4 vertex, 12 edge, 4 face) and 12
    more for the bubbles (8 face and 4 interior).

    Comparison of the two versions (diag_kmv3 files of each version, with the
    full matrices and the basis at 35 points of a lattice inside the cell).
    The nodes, the quadrature and the mass matrix are identical. Both
    stiffness matrices are symmetric with zero row sums, but the 2026.10.0 one
    is much larger (largest entry 103.6 against 6.2), and every block except
    the interior-interior one changes. In both versions the basis sums to one
    and interpolates every polynomial of degree 3 or less exactly (to 1e-15),
    so both spaces contain the cubics. Between the nodes, the vertex, edge and
    face basis functions differ by up to 0.33, 1.17 and 2.36, and the interior
    ones are identical. A nodal basis is unique for a given space and given
    nodes, so the 2026.10.0 space itself differs, in its bubble enrichment.
    It is also not symmetric: a swap P of two axes maps the cell and the
    nodes onto themselves, so phi_i(P x) = phi_P(i)(x) must hold; it holds to
    1e-15 with 2025.4.1, and fails by 2.48 with 2026.10.0 (vertex, edge and
    face functions; the interior ones keep it).
    """
    if is_firedrake_new():
        request.applymarker(KMV3_TETRAHEDRON_XFAIL)

    cell = FIAT.ufc_simplex(3)
    element = FIAT.KongMulderVeldhuizen(cell, 3)
    rule = make_quadrature(cell, 3, "KMV")
    points = np.asarray(rule.point_set.points)
    weights = np.asarray(rule.weights)
    nodes = np.array([list(node.get_point_dict())[0] for node in element.dual_basis()])

    tab = element.tabulate(1, points)
    phi = tab[(0, 0, 0)]
    grads = np.array([tab[key] for key in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    mass = np.einsum("ip,p,jp->ij", phi, weights, phi)
    stiffness = np.einsum("dip,p,djp->ij", grads, weights, grads)
    diagonal = np.diag(stiffness)

    # Current stiffness diagonal against the recorded versions
    recorded = {
        version: np.array(values.split(), dtype=float)
        for version, values in KMV3_STIFFNESS_DIAGONAL.items()
    }
    entity = [""] * element.space_dimension()
    names = {0: "vertex", 1: "edge", 2: "face", 3: "interior"}
    for dim, dofs in element.entity_dofs().items():
        for ids in dofs.values():
            for i in ids:
                entity[i] = names[dim]
    SpyroEnsemble.print(
        "\nStiffness diagonal of the degree 3 KMV tetrahedron\n"
        + f"{'Node':>4} {'Entity':>8} {'Current':>10}"
        + "".join(f" {version:>10}" for version in recorded)
    )
    for i, value in enumerate(diagonal):
        SpyroEnsemble.print(
            f"{i:>4} {entity[i]:>8} {value:>10.4f}"
            + "".join(f" {values[i]:>10.4f}" for values in recorded.values())
        )
    matches = []
    for version, values in recorded.items():
        diff = np.abs(diagonal - values)
        SpyroEnsemble.print(
            f"Max |current - {version}|: {diff.max():.3e}, at nodes "
            f"{np.flatnonzero(diff > KMV3_STIFFNESS_ATOL[version]).tolist()}"
        )
        if np.allclose(diagonal, values, rtol=0.0, atol=KMV3_STIFFNESS_ATOL[version]):
            matches.append(version)
    SpyroEnsemble.print(f"Current stiffness diagonal matches: {matches or 'none'}")

    # Basis at 35 points of a lattice inside the cell, between the nodes
    lattice = np.asarray(cell.make_points(3, 0, 8))
    phi_lattice = element.tabulate(0, lattice)[(0, 0, 0)]

    # Interpolation of every polynomial of degree 3 or less from its nodal values
    exponents = [e for e in itertools.product(range(4), repeat=3) if sum(e) <= 3]
    cubic_error = max(
        np.abs(
            np.prod(nodes ** np.array(e), axis=1) @ phi_lattice
            - np.prod(lattice ** np.array(e), axis=1)
        ).max()
        for e in exponents
    )
    SpyroEnsemble.print(
        f"Interpolation error of the {len(exponents)} monomials of degree 3 or less "
        f"at the lattice: {cubic_error:.2e}"
    )

    # Symmetry of the basis: a swap P of two axes maps the cell and the nodes
    # onto themselves, so phi_i(P x) = phi_P(i)(x)
    entity = np.array(entity)
    symmetry_error = 0.0
    for swap in ((1, 0, 2), (0, 2, 1), (2, 1, 0)):
        perm = [
            np.flatnonzero(np.all(np.isclose(nodes, node[list(swap)]), axis=1))[0]
            for node in nodes
        ]
        phi_swapped = element.tabulate(0, lattice[:, list(swap)])[(0, 0, 0)]
        diff = np.abs(phi_swapped - phi_lattice[perm]).max(axis=1)
        symmetry_error = max(symmetry_error, diff.max())
        SpyroEnsemble.print(
            f"Swap {swap}: max |phi_i(P x) - phi_P(i)(x)| {diff.max():.2e}, by node: "
            + ", ".join(
                f"{name} {diff[entity == name].max():.2e}" for name in names.values()
            )
        )

    # Same in both versions: nodes, quadrature, lumped mass and nodal basis
    assert element.space_dimension() == len(nodes) == len(weights) == 32
    dist = np.linalg.norm(nodes[:, None, :] - points[None, :, :], axis=2)
    assert np.all(dist.min(axis=1) < 1e-12), "✗ Quadrature points are not the nodes"
    assert np.allclose(mass, np.diag(np.diag(mass)), atol=1e-14), "✗ Mass not lumped"
    assert np.allclose(np.sort(np.diag(mass)), np.sort(weights)), "✗ Mass weights"
    assert np.allclose(element.tabulate(0, nodes)[(0, 0, 0)], np.eye(len(nodes)))
    assert (
        cubic_error < 1e-12
    ), f"✗ The basis does not reproduce the cubics: error {cubic_error:.2e}"

    # Differs with 2026.10.0: symmetry of the basis, and symmetry and values of
    # the stiffness diagonal
    assert symmetry_error < 1e-12, (
        "✗ The basis is not symmetric under the swaps of two axes, which map the "
        f"cell and the nodes onto themselves: error {symmetry_error:.2e}"
    )
    assert np.allclose(diagonal[1:4], diagonal[1], rtol=1e-10), (
        "✗ Stiffness entries of vertices 1, 2 and 3 differ, although a "
        f"permutation of the axes exchanges them: {diagonal[1:4]}"
    )
    reference = recorded["2025.4.1"]
    assert np.allclose(diagonal, reference, rtol=0.0, atol=1e-8), (
        "✗ Stiffness diagonal differs from Firedrake 2025.4.1 by up to "
        f"{np.abs(diagonal - reference).max():.3e} (sum {diagonal.sum():.4f} "
        f"against {reference.sum():.4f}); it matches: {matches or 'none'}"
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_spectral_element_reference_stiffness(dimension: int) -> None:
    """Check the spectral elements of the Q cases on the reference cell.

    This is the counterpart of test_kmv3_tetrahedron_reference_stiffness for
    the elements of the Q cases of test_infinite_model_abc, whose time steps
    are the same with Firedrake 2025.4.1 and 2026.10.0: the degree 4
    quadrilateral (Q-2D) and the degree 3 quadrilateral x interval element of
    the extruded meshes (Q-3D), both spectral (Gauss-Lobatto-Legendre). They
    are built with spyro's ``create_function_space`` and ``quadrature_rules``
    on one unit cell, which is their reference cell, and the mass and
    stiffness matrices are assembled with that quadrature. Run it with
    ``pytest -s`` to print the stiffness diagonal by group of dofs next to the
    values recorded with 2025.4.1 (SPECTRAL_STIFFNESS_DIAGONAL).

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.

    Returns
    -------
    None

    Notes
    -----
    The dofs are grouped by their distances to the faces of the cell, sorted,
    so the groups do not depend on the dof numbering: in 3D, the 8 vertex
    dofs, the 24 dofs on the edges, the 24 on the faces and the 8 interior
    ones. A permutation of the axes or a reflection maps the cell onto itself
    and keeps these distances, so the stiffness diagonal must be the same in
    each group; the degree 3 KMV tetrahedron of Firedrake 2026.10.0 breaks the
    corresponding symmetry. The whole matrices must also be symmetric in that
    sense: a swap of two axes or a reflection x -> 1 - x maps the cell onto
    itself and permutes the dofs, which must leave the mass and stiffness
    matrices unchanged. The space must reproduce every monomial of degree 4 (2D)
    or 3 (3D) or less in each variable: their nodal interpolants, all in one
    vector function, must have zero L2 error over the cell. The test also
    requires a diagonal mass matrix and the recorded values, which are rational
    numbers (0.7, 80/27, ... in 2D; 13/144, 155/432, 575/432 and 625/144 in
    3D).

    Results. Both cases pass with Firedrake 2025.4.1 (4.9 s) and 2026.10.0
    (4.6 s), with the same value in every group and the same values in both
    versions, so these elements did not change, unlike the degree 3 KMV
    tetrahedron. With both versions the matrices change by at most 5.3e-15
    under the symmetries of the cell, and the 25 (2D) and 64 (3D) monomials
    are reproduced to 9.3e-16 and 1.1e-15.
    """
    degree = 4 if dimension == 2 else 3  # as in wave_dict
    base = fire.UnitSquareMesh(1, 1, quadrilateral=True)
    mesh = base if dimension == 2 else fire.ExtrudedMesh(base, 1, layer_height=1.0)
    V = create_function_space(mesh, "spectral_quadrilateral", degree)
    quad_rule = quadrature_rules(V)[0]
    u, v = fire.TrialFunction(V), fire.TestFunction(V)
    mass = fire.assemble(u * v * fire.dx(**quad_rule)).petscmat
    stiffness = fire.assemble(
        fire.inner(fire.grad(u), fire.grad(v)) * fire.dx(**quad_rule)
    ).petscmat
    coords = fire.Function(fire.VectorFunctionSpace(mesh, V.ufl_element()))
    coords.interpolate(fire.SpatialCoordinate(mesh))

    # Stiffness diagonal by group of dofs with the same distances to the faces
    groups = {}
    for x, value in zip(coords.dat.data_ro, stiffness.getDiagonal().getArray()):
        key = tuple(sorted(float(round(min(c, 1.0 - c), 6)) for c in x))
        groups.setdefault(key, []).append(value)
    recorded = SPECTRAL_STIFFNESS_DIAGONAL[dimension]
    SpyroEnsemble.print(
        f"\nStiffness diagonal of {V.ufl_element()}\n"
        + f"{'Distances to the faces':>26} {'Dofs':>5} {'Min':>12} {'Max':>12} "
        + f"{'2025.4.1':>12}"
    )
    for key in sorted(groups):
        values = np.array(groups[key])
        SpyroEnsemble.print(
            f"{str(key):>26} {len(values):>5} {values.min():>12.8f} "
            f"{values.max():>12.8f} {recorded.get(key, np.nan):>12.8f}"
        )

    # Dense copies; the converted matrices must stay alive while they are read
    dense = {}
    for name, matrix in (("mass", mass), ("stiffness", stiffness)):
        converted = matrix.convert("dense")
        dense[name] = converted.getDenseArray().copy()

    # Symmetry of the matrices: a swap of two axes or a reflection x -> 1 - x
    # maps the cell onto itself and permutes the dofs, which must leave the
    # mass and stiffness matrices unchanged
    x_dofs = coords.dat.data_ro
    index = {tuple(np.round(x, 10)): i for i, x in enumerate(x_dofs)}
    transforms = [
        ("swap", axes) for axes in itertools.combinations(range(dimension), 2)
    ]
    transforms += [("reflection", (axis,)) for axis in range(dimension)]
    symmetry_error = 0.0
    for kind, axes in transforms:
        y_dofs = x_dofs.copy()
        if kind == "swap":
            y_dofs[:, list(axes)] = y_dofs[:, list(axes[::-1])]
        else:
            y_dofs[:, axes[0]] = 1.0 - y_dofs[:, axes[0]]
        perm = [index[tuple(np.round(y, 10))] for y in y_dofs]
        errors = {
            name: np.abs(matrix[np.ix_(perm, perm)] - matrix).max()
            for name, matrix in dense.items()
        }
        symmetry_error = max(symmetry_error, *errors.values())
        SpyroEnsemble.print(
            f"{kind} {axes}: change of the mass {errors['mass']:.2e}, "
            f"of the stiffness {errors['stiffness']:.2e}"
        )

    # Nodal interpolation of every monomial up to the degree in each variable,
    # all in one vector function, and its L2 error over the cell
    exponents = list(itertools.product(range(degree + 1), repeat=dimension))
    xs = fire.SpatialCoordinate(mesh)
    exact = fire.as_vector(
        [np.prod([xs[k] ** e[k] for k in range(dimension)]) for e in exponents]
    )
    W = fire.VectorFunctionSpace(mesh, V.ufl_element(), dim=len(exponents))
    interpolant = fire.Function(W)
    interpolant.dat.data[:] = np.array(
        [np.prod(x_dofs ** np.array(e), axis=1) for e in exponents]
    ).T
    poly_error = np.sqrt(
        fire.assemble(fire.inner(interpolant - exact, interpolant - exact) * fire.dx)
    )
    SpyroEnsemble.print(
        f"L2 error of the nodal interpolants of the {len(exponents)} monomials of "
        f"degree {degree} or less in each variable: {poly_error:.2e}"
    )

    # Lumped mass, symmetric matrices, polynomials and the same groups as recorded
    off_diagonal = dense["mass"] - np.diag(np.diag(dense["mass"]))
    assert np.abs(off_diagonal).max() < 1e-12, "✗ Mass matrix not diagonal"
    assert (
        symmetry_error < 1e-12
    ), f"✗ Mass or stiffness changes under a symmetry of the cell: {symmetry_error:.2e}"
    assert (
        poly_error < 1e-12
    ), f"✗ The space does not reproduce the monomials: error {poly_error:.2e}"
    assert sorted(groups) == sorted(recorded), "✗ Groups of dofs differ"

    # Symmetry within each group and the recorded values
    for key, values in groups.items():
        assert np.allclose(values, values[0], rtol=1e-10), (
            f"✗ Stiffness entries of the dofs {key} differ, although symmetries "
            f"of the cell exchange them: {values}"
        )
        assert np.isclose(np.mean(values), recorded[key], rtol=1e-10), (
            f"✗ Stiffness entry of the dofs {key} is {np.mean(values)}, "
            f"recorded {recorded[key]} with Firedrake 2025.4.1"
        )
