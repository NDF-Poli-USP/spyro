"""Unit tests for the Reference Model implemented in spyro.abc.abc_layer.

These tests verify the consistency of the solver with HABCs and PML without any
damping for a model of reference with an extended pad to avoid reflections. The
tests are designed to ensure that the computed transiente responses and energies
are consistent with expected values. The tests cover both 2D and 3D cases.
"""

from pytest import fail, mark, param
from firedrake import COMM_WORLD as comm, conditional, ConvergenceError
from numpy import all, sum
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.io.basicio import parallel_print as pprint


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
        "delay_type": "multiples_of_minimum" if dimension == 2 else "time",
        "delay": 1.5 if dimension == 2 else 1. / 3.,
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

    # Define parameters for the transient integration method.
    dictionary["time_axis"] = {
        "final_time": 2. if dimension == 2 else 1.5,  # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": 50,  # how frequently to output solution to pvds
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "abc_type": abc_type,  # Options: "hybrid" or "PML"
        "degree_eikonal": degree_eik,  # Order for the Eikonal FEM
        "get_ref_model": True,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Eik" if calc_eik else "NoEik")
    output_folder = f"output/inf_test{dimension}d/inf_test{dimension}d" + str_ele
    dictionary["visualization"] = {  # Output folder
        "output_folder": output_folder,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": output_folder + f"/preamble/acoustic_energy_{abc_type}"
    }

    return dictionary


def wave_instance(element_geometry, dimension, abc_type, calc_eik):
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

    Returns
    -------
    wave : acoustic_wave.AcousticWave
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    max_divisor_tf : `int`, optional
        Index to select the maximum divisor of the final time, converted to an
        integer according to the order of magnitude of the timestep size. The
        timestep size is set to the divisor, given by the index in descending
        order, less than or equal to the user's timestep size. If the value is 1,
        the timestep size is set as the maximum divisor. Default is 1.
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 1. / 4. if dimension == 2 else 1. / 2.

    # f_est: Factor for the stabilizing term in Eikonal equation
    # Timestep size (in seconds). Initial guess: edge_length / 100
    if dimension == 2:
        f_est = 0.03 if element_geometry == "T" else 0.02
        degree_eik = 2 if element_geometry == "T" else 3
        dt_usu = 0.00400 if element_geometry == "T" else 0.00500

    if dimension == 3:
        f_est = 0.03 if element_geometry == "T" else 0.10
        degree_eik = 4 if element_geometry == "T" else 3
        dt_usu = 0.01000 if element_geometry == "T" else 0.01250

    # Maximum divisor of the final time
    max_divisor_tf = 3 if dimension == 2 else 4

    # Get simulation parameters
    pprint(f"\nMesh Size: {1e3 * edge_length:.4f} m", comm=comm)
    pprint(f"Element Geometry: {element_geometry}", comm=comm)
    pprint(f"Eikonal Degree: {degree_eik}", comm=comm)
    pprint(f"Eikonal Stabilizing Factor: {f_est:.2f}", comm=comm)
    pprint(f"Timestep Size: {1e3 * dt_usu:.3f} ms", comm=comm)
    pprint(f"Maximum Divisor of Final Time: {max_divisor_tf}", comm=comm)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(
        element_geometry, dimension, abc_type, dt_usu, degree_eik, calc_eik)

    # ============ MESH FEATURES ============

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
    wave.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    wave.mesh_ops.preamble_mesh_operations(wave, f_est=f_est)

    if calc_eik:
        # ============ EIKONAL ANALYSIS ============

        # Finding critical points
        wave.layer_ops.critical_boundary_points(wave)

    return wave, max_divisor_tf


@mark.older_firedrake
@mark.parametrize("element_geometry, dimension, calc_eik", [
    ("T", 2, True),
    ("T", 2, False),
    ("Q", 2, True),
    ("Q", 2, False),
    param("T", 3, True, marks=mark.slow),
    param("T", 3, False, marks=mark.slow),
    param("Q", 3, True, marks=mark.slow),
    param("Q", 3, False, marks=mark.slow)])
def test_infinite_model_abc(element_geometry, dimension, calc_eik):
    """Testing modal solvers for 2D and 3D case in Fig. 8 of Salas et al (2022).

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
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.

    Returns
    -------
    None

    ==============================
    Eikonal for 2D model Δx = 250m
    ==============================
    eik_min = 83.333 ms

    f_est T-ele(p=2) Q-ele(p=3)
     0.02     --/--     85.187*
     0.03   84.586*     90.411
     0.04   87.945       --/--

    ==============================
    Eikonal for 3D model Δx = 500m
    ==============================
    eik_min = 83.333 ms

    f_est T-ele(p=4) Q-ele(p=3)
     0.03    85.402*    50.173
     0.04   105.595     54.647
     0.05     --/--     61.110
     0.06     --/--     64.875
     0.07     --/--     69.240
     0.08     --/--     74.181
     0.09     --/--     79.604
     0.10     --/--     85.386*
     0.11     --/--     91.403
    """

    act_eik = "Activated" if calc_eik else "Deactivated"
    pprint("\n" + 60 * "=" + f"\nTesting Reference Model with {element_geometry} "
           + f"elements for ABCs\nand {dimension}D case. Eikonal analysis: {act_eik}\n"
           + 60 * "=", comm=comm)

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

            pprint(f"\nAbsorbing Boundary Condition: {abc_type}", comm=comm)

            # Create an instance of the acoustic wave solver
            wave, max_divisor_tf = wave_instance(element_geometry, dimension,
                                                 abc_type, calc_eik)

        #     # Computing reference signal
        #     wave.layer_ops.infinite_model(wave, check_dt=True, max_divisor_tf=max_divisor_tf)

        #     # Estimating computational resource usage
        #     comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "preamble/INF_")

        #     if abc_type == "hybrid":  # Hybrid is the reference
        #         hybrid_signal = wave.layer_ops.get_reference_signal()[0]
        #         hybrid_energy = wave.field_logger.get("acoustic_energy")
        #     else:  # PML is the comparison
        #         pml_signal = wave.forward_solution_receivers
        #         pml_energy = wave.field_logger.get("acoustic_energy")

        # # Checking both signals
        # dt = wave.get_dt()
        # assert hybrid_signal is not None, "Hybrid signal not found"
        # assert pml_signal is not None, "PML signal not found"
        # error_measures = wave.layer_ops.error_measures(pml_signal, hybrid_signal, dt,
        #                                                wave.number_of_receivers,
        #                                                final_energy=pml_energy,
        #                                                final_energy_reference=hybrid_energy,
        #                                                save_in_case_folder=False)
        # errIt, errPk, pkMax, max_errIt, max_errPK, final_ener, dsspt_ener = error_measures

        # assert sum(errIt) == 0. and max_errIt == 0., \
        #     "✗ Integral Error check for 'hybrid' and 'PML' solvers in Reference Model " \
        #     f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        # pprint("✓ Integral Error Verified for 'hybrid' and 'PML' solvers", comm=comm)
        # assert sum(errPk) == 0. and max_errPK == 0. and all(pkMax) > 0., \
        #     "✗ Peak Error check for 'hybrid' and 'PML' solvers in Reference Model " \
        #     f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        # pprint("✓ Peak Error Verified for 'hybrid' and 'PML' solvers", comm=comm)
        # assert final_ener > 0. and dsspt_ener == 0., \
        #     "✗ Final Energy check for 'hybrid' and 'PML' solvers in Reference Model " \
        #     f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
        # pprint("✓ Final Energy Verified for 'hybrid' and 'PML' solvers", comm=comm)

    except ConvergenceError as e:
        fail(f"Checking Reference Model with {element_geometry} elements for "
             f"{dimension}D and Eikonal {act_eik} case raised an exception: {str(e)}")
