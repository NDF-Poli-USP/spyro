from pytest import fail, mark, param
from firedrake import COMM_WORLD as comm, conditional, ConvergenceError
from numpy import isclose
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.io.basicio import parallel_print as pprint


def wave_dict(element_geometry, dimension, dt_usu, get_ref_model):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    dt_usu: `float`
        Time step of the simulation
    get_ref_model : `bool`
        If `True`, the infinite ou refercne model is created. If `False`, Non-Reflective
        BCs (NRBCs) are applied to the model.

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
        "abc_type": "nrbc",  # Activate NRBC
        "get_ref_model": get_ref_model,  # If `True`, the infinite model is created
    }

    # Define parameters for visualization
    output_folder = f"output/nrbc_test{dimension}d/nrbc_test{dimension}d{element_geometry}"
    dictionary["visualization"] = {  # Output folder
        "output_folder": output_folder,
    }

    return dictionary


def wave_instance(element_geometry, dimension, get_ref_model):
    """Create an instance of the acoustic wave solver.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    get_ref_model : `bool`
        If `True`, the infinite ou refercne model is created. If `False`, Non-Reflective
        BCs (NRBCs) are applied to the model.

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
    edge_length = 0.25 if dimension == 2 else 0.5

    # Timestep size (in seconds). Initial guess: edge_length / 100
    if dimension == 2:
        dt_usu = 0.00400 if element_geometry == "T" else 0.00500

    if dimension == 3:
        dt_usu = 0.01000 if element_geometry == "T" else 0.01250

    # Maximum divisor of the final time
    max_divisor_tf = 3 if dimension == 2 else 4

    # Get simulation parameters
    pprint(f"\nMesh Size: {1e3 * edge_length:.4f} m", comm=comm)
    pprint(f"Element Geometry: {element_geometry}", comm=comm)
    pprint(f"Timestep Size: {1e3 * dt_usu:.3f} ms", comm=comm)
    pprint(f"Maximum Divisor of Final Time: {max_divisor_tf}", comm=comm)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_geometry, dimension, dt_usu, get_ref_model)

    # ============ MESH FEATURES ============

    # Create the acoustic wave object with HABCs
    wave = AcousticWave(dictionary=dictionary)

    # Mesh
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(wave.mesh_x < 0.5, 3.0, 1.5)
    wave.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    wave.mesh_ops.preamble_mesh_operations(wave)

    return wave, max_divisor_tf


@mark.parametrize("element_geometry, dimension", [("T", 2),
                                                  ("Q", 2),
                                                  ("T", 3),
                                                  ("Q", 3),
                                                  ])
def test_nrbc(element_geometry, dimension):
    """Testing NRBCs for 2D and 3D case in Fig. 8 of Salas et al (2022).




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

    Returns
    -------
    None
    """

    pprint("\n" + 60 * "=" + f"\nTesting NRBCs with {element_geometry} "
           + f"elements and {dimension}D case\n"
           + 60 * "=", comm=comm)

    # ============ REFERENCE MODEL ============

    get_ref_model = True

    try:

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Create an instance of the acoustic wave solver
        wave, max_divisor_tf = wave_instance(element_geometry, dimension, get_ref_model)

    #     # Computing reference get_reference_signal
    #     wave.layer_ops.infinite_model(wave, check_dt=True,
    #                                   max_divisor_tf=max_divisor_tf)

    #     receivers_reference, receivers_ref_fft = wave.layer_ops.get_reference_signal()

    #     # Estimating computational resource usage
    #     comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "preamble/INF_")

    #     if abc_type == "hybrid":
    #         hybrid_signal = receivers_reference
    #         hybrid_energy = wave.field_logger.get("acoustic_energy")
    #     else:
    #         pml_signal = receivers_reference
    #         pml_energy = wave.field_logger.get("acoustic_energy")

    #     # Checking both signals
    #     assert hybrid_signal is not None, "Hybrid signal not found"
    #     assert pml_signal is not None, "PML signal not found"

    #     dt = wave.get_dt()
    #     error_measures = wave.layer_ops.error_measures(pml_signal, hybrid_signal, dt,
    #                                                    wave.number_of_receivers,
    #                                                    final_energy=pml_energy,
    #                                                    final_energy_reference=hybrid_energy,
    #                                                    save_in_case_folder=False)
    #     errIt, errPk, pkMax, max_errIt, max_errPK, final_ener, dsspt_ener = error_measures

    #     assert sum(errIt) == 0. and max_errIt == 0., \
    #         "✗ Integral Error check for 'hybrid' and 'PML' solvers in Reference Model " \
    #         f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
    #     pprint("✓ Integral Error Verified for 'hybrid' and 'PML' solvers", comm=comm)
    #     assert sum(errPk) == 0. and max_errPK == 0. and all(pkMax) > 0., \
    #         "✗ Peak Error check for 'hybrid' and 'PML' solvers in Reference Model " \
    #         f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
    #     pprint("✓ Peak Error Verified for 'hybrid' and 'PML' solvers", comm=comm)
    #     assert final_ener > 0. and dsspt_ener == 0., \
    #         "✗ Final Energy check for 'hybrid' and 'PML' solvers in Reference Model " \
    #         f"{dimension}D with {element_geometry} elements and Eikonal {act_eik} case."
    #     pprint("✓ Final Energy Verified for 'hybrid' and 'PML' solvers", comm=comm)

    except ConvergenceError as e:
        fail(f"Checking NRBCs with {element_geometry} elements for "
             f"{dimension}D raised an exception: {str(e)}")
