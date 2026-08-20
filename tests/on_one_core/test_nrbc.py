from pytest import fail, mark, param
from firedrake import COMM_WORLD as comm, conditional, ConvergenceError
from numpy import isclose, load
from spyro.io.basicio import parallel_print as pprint
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost
from spyro.utils.typing import BoundaryConditionsType


def wave_dict(element_geometry, dimension, dt_usu, get_ref_model, degree_eik, calc_eik):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    dt_usu: `float`
        Time step of the simulation.
    get_ref_model : `bool`
        If `True`, the infinite ou refercne model is created. If `False`, Non-Reflective
        BCs (NRBCs) are applied to the model.
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
        "type": "automatic" if calc_eik else "spatial",
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
    if calc_eik:
        source_locations = ([(-length_z / 2., length_x / 4.)] if dimension == 2
                            else [(-length_z / 2., length_x / 4., length_y / 2.)])
    else:
        source_locations = ([(-length_z / 2., length_x / 4.),
                             (-length_z / 4., 5 * length_x / 8.),
                             (-3 * length_z / 4., 5 * length_x / 8.)] if dimension == 2
                            else [(-length_z / 2., length_x / 4., length_y / 2.),
                                  (-length_z / 4., 5 * length_x / 8., length_y / 4.),
                                  (-3 * length_z / 4., 5 * length_x / 8., length_y / 4.)])

    dictionary["acquisition"] = {
        "source_locations": source_locations,
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
        "output_frequency": 15,  # how frequently to output solution to pvds
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "abc_type": "nrbc",  # Activate NRBC
        "degree_eikonal": degree_eik,  # Order for the Eikonal FEM
        "get_ref_model": get_ref_model,  # If `True`, the infinite model is created
    }

    # Define parameters for visualization
    str_ele = element_geometry + "_" + ("Eik" if calc_eik else "NoEik")
    output_folder = f"output/nrbc_test{dimension}d/nrbc_test{dimension}d" + str_ele
    dictionary["visualization"] = {  # Output folder
        "output_folder": output_folder,
        "acoustic_energy": True,  # Activate energy calculation
    }

    return dictionary


def wave_instance(element_geometry, dimension, calc_eik, get_ref_model):
    """Create an instance of the acoustic wave solver.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    calc_eik : `bool`
        If `True`, eikonal analysis is performed; otherwise, it is skipped.
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
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 1. / 4.

    # f_est: Factor for the stabilizing term in Eikonal equation
    # Timestep size (in seconds). Initial guess: edge_length / 100
    if dimension == 2:
        f_est = 0.03 if element_geometry == "T" else 0.02
        degree_eik = 2 if element_geometry == "T" else 3
        dt_usu = 0.00250 if element_geometry == "T" else 0.00320

    if dimension == 3:
        f_est = 0.20 if element_geometry == "T" else 0.10
        degree_eik = 1
        dt_usu = 0.00300 if element_geometry == "T" else 0.00400

    # Maximum divisor of the final time
    max_divisor_tf = 5 if dimension == 2 else 7

    # Get simulation parameters
    pprint(f"\nMesh Size: {1e3 * edge_length:.4f} m", comm=comm)
    pprint(f"Element Geometry: {element_geometry}", comm=comm)
    pprint(f"Eikonal Degree: {degree_eik}", comm=comm)
    pprint(f"Eikonal Stabilizing Factor: {f_est:.2f}", comm=comm)
    pprint(f"Timestep Size: {1e3 * dt_usu:.3f} ms", comm=comm)
    pprint(f"Maximum Divisor of Final Time: {max_divisor_tf}", comm=comm)

    # Create dictionary with parameters for the model
    dictionary = wave_dict(
        element_geometry, dimension, dt_usu, get_ref_model, degree_eik, calc_eik)

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
        wave.nrbc_ops.critical_boundary_points(wave)

    return wave, max_divisor_tf, dictionary


@mark.parametrize("element_geometry, dimension, calc_eik", [
    ("T", 2, True),
    # ("Q", 2, True),
    # ("T", 3, True),
    # ("Q", 3, True),
    # ("T", 2, False),
    # ("Q", 2, False),
    # ("T", 3, False),
    # ("Q", 3, False),
])
def test_nrbc(element_geometry, dimension, calc_eik):
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

    =================================
    Eikonal for 3D model Δx = 333.33m
    =================================
    eik_min = 83.333 ms

    f_est   T-ele   Q-ele
     0.02  60.886   --/--
     0.03  61.715  69.911
     0.04  62.513  71.755
     0.05  63.256  73.351
     0.10  67.891  81.172
     0.11   --/--  82.943*
     0.12   --/--  84.783
     0.15  75.965   --/--
     0.18  81.912   --/--
     0.19  83.969*  --/--
     0.20  86.037   --/--

    f_est T-ele(p=1) Q-ele(p=1)
     0.09     --/--     78.507
     0.10     --/--     87.870*
     0.11     --/--     97.539
     0.19    80.596      --/--
     0.20    83.374*     --/--
     0.21    86.079      --/--
    """

    act_eik = "Activated" if calc_eik else "Deactivated"
    pprint("\n" + 75 * "=" + f"\nTesting NRBCs with {element_geometry} elements and "
           + f"{dimension}D case. Eikonal analysis: {act_eik}\n" + 75 * "=", comm=comm)

    get_ref_model = True
    get_nrbc_model = True

    try:

        # ============ MODEL PARAMETERS AND EIKONAL ============

        # Create an instance of the acoustic wave solver
        wave, max_divisor_tf, dictionary = wave_instance(element_geometry, dimension,
                                                         calc_eik, get_ref_model)
        energy_name = "acoustic_ref_energy"
        energy_file = wave.path_save + "preamble/" + energy_name

        # ============ REFERENCE MODEL ============

        if get_ref_model:

            # Reference to resource usage
            tRef = comp_cost("tini")

            # Updating visualization dictionary with acoustic energy filename
            dictionary["visualization"].update({"acoustic_energy_filename": energy_file})

            # Computing reference signal
            wave.nrbc_ops.infinite_model(wave, check_dt=True, max_divisor_tf=max_divisor_tf)

            # Set model parameters for the NRBC scheme
            wave.abc_get_ref_model = False

            # Estimating computational resource usage
            comp_cost("tfin", tRef=tRef, user_name=wave.path_save + "preamble/INF_")

        # ============ NRBC SCHEME ============

        if get_nrbc_model:

            # Acquiring reference for signal and acoustic energy
            output_file = wave.abc_type.value + "_ref"
            receivers_reference, receivers_ref_fft, energy_reference = \
                wave.nrbc_ops.get_reference_signal(output_file=output_file,
                                                   get_energy_reference=True,
                                                   energy_reference_file=energy_name)
            final_energy_reference = energy_reference[-1]

            if get_ref_model:
                # Returning to the original mesh nad velocity profile
                wave.set_mesh(user_mesh=wave.mesh_original)
                wave.c = wave.mesh_ops.creating_velocity_profile(
                    wave.function_space, wave.initial_velocity_model, wave.path_save)[0]

            # Time step size for the transient response
            dt = wave.get_dt()

            for nrbc_type in [BoundaryConditionsType.HIGDON,
                              BoundaryConditionsType.SOMMERFELD]:

                # Tolerance for error measures
                tol_max_err = 0.20 if nrbc_type == BoundaryConditionsType.SOMMERFELD \
                    else 0.15
                tol_min_err = 0.003 if nrbc_type == BoundaryConditionsType.SOMMERFELD \
                    else 0.006

                # Critical source position
                crit_source = wave.nrbc_ops.eik_bnd[0][-1] if calc_eik else None

                # Reference to resource usage
                tRef = comp_cost("tini")

                # Updating NRBC type in the wave object
                wave.nrbc_ops.non_reflect_bc = nrbc_type
                energy_nrbc = wave.nrbc_ops.path_case_nrbc + "acoustic_energy"

                # Updating visualization dictionary with acoustic energy filename
                dictionary["visualization"].update({"acoustic_energy_filename": energy_nrbc})

                # Applying NRBCs on original domain boundary
                wave.nrbc_ops.nrbc_on_boundary(wave, source_coord=crit_source)

                # Solving the forward problem
                wave.forward_solve()

                # Acquiring final acoustic energy
                final_energy = wave.field_logger.get("acoustic_energy")
                energy = load(energy_nrbc + ".npy").T

                # Calculating error measures between the NRB and the reference models
                error_measures = wave.nrbc_ops.error_measures(
                    wave.forward_solution_receivers, receivers_reference, dt,
                    wave.number_of_receivers, final_energy=final_energy,
                    final_energy_reference=final_energy_reference)
                errIt, errPk, pkMax, max_errIt, \
                    max_errPK, final_ener, dsspt_ener = error_measures

                # Plotting the solution at receivers and the error measures
                wave.nrbc_ops.comparison_plots(wave, receivers_reference)

                # Estimating computational resource usage
                comp_cost("tfin", tRef=tRef, user_name=wave.nrbc_ops.path_case_nrbc)

                nrbc_str = wave.nrbc_ops.non_reflect_bc.value
                assert min(errIt) >= tol_min_err and max_errIt > tol_max_err, \
                    f"✗ Integral Error check for {nrbc_str} BC in Model {dimension}D " \
                    f"with {element_geometry} elements and Eikonal {act_eik} case."
                pprint(f"✓ Integral Error Verified for {nrbc_str} BC", comm=comm)
                assert min(errPk) >= tol_min_err and max_errPK > tol_max_err and all(pkMax) > 0., \
                    f"✗ Peak Error check for {nrbc_str} BC in Model {dimension}D " \
                    f"with {element_geometry} elements and Eikonal {act_eik} case."
                pprint(f"✓ Peak Error Verified for {nrbc_str} BC", comm=comm)
                assert final_ener > 0. and dsspt_ener > 0., \
                    f"✗ Final Energy check for {nrbc_str} BC in Model {dimension}D " \
                    f"with {element_geometry} elements and Eikonal {act_eik} case."
                pprint(f"✓ Final Energy Verified for {nrbc_str} BC", comm=comm)

    except ConvergenceError as e:
        fail(f"Checking NRBCs with {element_geometry} elements for "
             f"{dimension}D raised an exception: {str(e)}")
