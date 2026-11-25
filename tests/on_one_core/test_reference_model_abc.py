"""Unit tests for the Reference Model implemented in spyro.abc.abc_layer.

These tests verify the consistency of the solver with HABCs and PML without any
damping for a model of reference with an extended pad to avoid reflections. The
tests are designed to ensure that the computed transiente responses and energies
are consistent with expected values. The tests cover both 2D and 3D cases.
"""

from types import SimpleNamespace

import pytest
from firedrake import conditional, ConvergenceError
from numpy import all, finfo, float32, isclose, sum
from spyro.abc.abc_layer import ABCLayer
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.version_control import is_firedrake_new
from tests.on_one_core.test_eikonal import case_parameter
from tests.on_one_core.test_firedrake_elements import KMV3_TETRAHEDRON_XFAIL

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
    "f_est": {2: {"T": 0.03, "Q": 0.05}, 3: {"T": 0.68, "Q": 0.07}},
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

# Tolerance of the comparison of the hybrid and PML references: the machine
# epsilon of 32-bit floats. In serial both schemes perform the same operations
# and their errors are exactly zero. With spatial parallelism, runs on different
# partitions add in another order and their errors are round-off (1.3e-27)
ROUND_OFF_TOL = finfo(float32).eps


def wave_dict(element_geometry, dimension, abc_type, dt_usu, degree_eik, calc_eik):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    abc_type : `str`
        Type of the absorbing boundary condition. Options: "hybrid" or "PML".
    dt_usu: `float`
        Time step of the simulation.
    degree_eik : `int`
        Finite element order for the Eikonal equation.
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.

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
    dictionary = wave_dict(
        element_geometry, dimension, abc_type, dt_usu, degree_eik, calc_eik
    )

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
    damping: their errors must be zero up to ``ROUND_OFF_TOL``, the machine
    epsilon of 32-bit floats (round-off of runs on different partitions).

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
    value is 33 % below the theoretical one. Values with Firedrake 2025.4.1.

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
     0.62 55.721  0.05  48.938
     0.64 55.839  0.06  63.319
     0.66 55.909  0.07  78.381*
     0.67 55.927  0.08  93.750
     0.68 55.934* 0.09 109.148
     0.70 55.919  0.10 124.374

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
    T-3D    15.945       11.197         11.137    15.0  12.5  12.0  10.0
    Q-3D    26.688       22.640         22.335    25.0  20.0  15.0  12.5

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
    times smaller, so the largest eigenvalue is about 123 times larger, and the
    four T-3D cases fail:

    Version   ANALYTICAL  Exact (orig.)  Divisors <= ANALYTICAL  'ANA'  'ARN'
    2025.4.1    15.945       11.197      15.0  12.5  12.0  10.0  10.0   10.0
    2026.10.0    1.885        1.010      1.50  1.25  1.20  1.00   1.0    1.0

    Both methods select the fourth divisor (1.0 ms), just below the exact
    stable time step of the original mesh (1.010 ms), and the forward solves
    stop after 0.1 s of simulation. The minimum Eikonal at f_est = 0.68 is
    56.779 ms (55.934), also the largest from f_est = 0.66 to 0.70 with
    2026.10.0. The cause is the degree 3 KMV tetrahedron of FIAT.
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
    (test_spectral_element_reference_stiffness in test_firedrake_elements
    shows that the latter two are also identical on the reference cell). Other
    degrees of the KMV tetrahedron were not checked. Every spyro simulation
    with lumped degree 3 tetrahedra is affected with 2026.10.0, not only this
    test. With newer Firedrake (``is_firedrake_new()``, version 2026.4 or later;
    only 2026.10.0 was checked), the four T-3D cases are therefore expected
    failures (KMV3_TETRAHEDRON_XFAIL, strict): they still run, and are reported
    as soon as they pass, for example once the element is corrected upstream.
    With the mark, 2026.10.0 reports the four T-3D cases as xfailed: their
    forward solves stop after 0.1 s of simulation.
    test_kmv3_tetrahedron_reference_stiffness in test_firedrake_elements
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

            # Estimating computational resource usage
            comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "preamble/INF_")

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

            if abc_type == "hybrid":  # Hybrid is the reference
                hybrid_signal = receivers_reference
                hybrid_energy = wave.simulation_logger.get("acoustic_energy")
            else:  # PML is the comparison
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

        assert sum(errIt) <= ROUND_OFF_TOL and max_errIt <= ROUND_OFF_TOL, (
            "✗ Integral Error check for 'hybrid' and 'PML' solvers in Reference Model "
            f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        )
        SpyroEnsemble.print("✓ Integral Error Verified for 'hybrid' and 'PML' solvers")
        assert (
            sum(errPk) <= ROUND_OFF_TOL
            and max_errPK <= ROUND_OFF_TOL
            and all(pkMax) > 0.0
        ), (
            "✗ Peak Error check for 'hybrid' and 'PML' solvers in Reference Model "
            f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        )
        SpyroEnsemble.print("✓ Peak Error Verified for 'hybrid' and 'PML' solvers")
        assert final_ener > 0.0 and abs(dsspt_ener) <= ROUND_OFF_TOL, (
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


@pytest.mark.parametrize("c_max", [C_BND_MAX, 4.5, 6.0])
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
def test_layer_infinite_model_without_eikonal(
    dimension: int, sources: list, c_max: float
) -> None:
    """Check that no reflection returns before the final time without Eikonal.

    Without Eikonal analysis, the earliest time at which the wave reaches the
    absorbing boundaries is bounded below by the distance from the nearest
    source to the nearest of them (the bottom z = -1 and the lateral
    boundaries; z = 0 is the free surface) over ``c_max``, since the interior
    can be faster than the boundary. A wave reflected by the boundary of the
    extension returns to the original boundary ``2 * pad / c_bnd_max`` later,
    which must not be before the final time, and the extension must not be
    longer than needed by more than ``lmin``.

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    sources : `list`
        Source locations (z, x) or (z, x, y) in km.
    c_max : `float`
        Maximum velocity in km/s of the original domain.

    Returns
    -------
    None
    """
    # Distance from the nearest source to the nearest absorbing boundary
    dist = min(
        min([1.0 + src[0]] + [min(coord, 1.0 - coord) for coord in src[1:]])
        for src in sources
    )
    t_bnd = dist / c_max
    needed = C_BND_MAX * (FINAL_TIME - t_bnd) / 2.0

    pad = abc_layer(dimension).layer_infinite_model(
        LMIN, C_BND_MAX, c_max, FINAL_TIME, source_locations=sources
    )

    assert t_bnd + 2.0 * pad / C_BND_MAX >= FINAL_TIME - 1e-12
    assert needed - 1e-12 <= pad < needed + LMIN
    assert isclose(pad / LMIN, round(pad / LMIN))


@pytest.mark.parametrize("eikmin", [0.084586, 0.055934])
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
        LMIN, C_BND_MAX, C_BND_MAX, FINAL_TIME, source_locations=[(-0.9, 0.5)]
    )

    assert eikmin + 2.0 * pad / C_BND_MAX >= FINAL_TIME - 1e-12
    assert needed - 1e-12 <= pad < needed + LMIN


def test_layer_infinite_model_c_max_below_boundary() -> None:
    """Check that a domain maximum below the boundary maximum is rejected.

    ``c_max`` and ``c_bnd_max`` are rounded with the same tolerance, so the
    maximum velocity of the domain cannot be smaller than that of its boundary.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        abc_layer(2).layer_infinite_model(
            LMIN, C_BND_MAX, 2.0, FINAL_TIME, source_locations=[(-0.5, 0.25)]
        )


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
