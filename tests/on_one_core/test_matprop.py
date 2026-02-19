import pytest
import firedrake as fire
import warnings
import numpy as np
from os import getcwd
from spyro.io.basicio import create_segy
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(domain_dim, tf_usu, dt_usu):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    domain_dim : `list`
        List containing the domain dimensions [Lz, Lx, Ly] in km
    tf_usu : `float`
        Final time of the simulation
    dt_usu: `float`
        Time step of the simulation

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "Q",
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 3,  # p order p<=3 for 3D
        "dimension": 3,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    Lz, Lx, Ly = domain_dim  # in km
    dictionary["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with
    # an explosive source (moment source) that has a peak frequency of 5 Hz
    # injected at a specified point of the mesh. We also specify to record
    # the solution at the corners of the domain to verify the NRBC efficiency.
    dictionary["acquisition"] = {
        "source_locations": [(-Lz / 2., Lx / 2., Ly / 2.)],
        "frequency": 5.0,  # in Hz
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.), (0., 0., 0),
                               (0., Lx, 0.), (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1.5 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": tf_usu,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
    }

    return dictionary


@pytest.fixture(scope="function")
def wave_instance():
    '''
    Create an instance of the isotropic wave solver.

    Wave_obj : `wave.IsotropicWave`
        An instance of the IsotropicWave class
    '''

    # Domain dimensions
    domain_dim = [0.24, 0.56, 0.16]  # in km

    # Final Time in s
    tf_usu = 2.

    # Number of timesteps
    steps = 200

    # Time step of the simulation
    dt_usu = round(tf_usu / steps, 6)

    # Mesh size in km
    edge_length = 0.040

    # Create dictionary with parameters for the model
    dictionary = wave_dict(domain_dim, tf_usu, dt_usu)

    # Create a wave object
    Wave_obj = IsotropicWave(dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    return Wave_obj


def test_constant_mat_prop(wave_instance):
    '''
    Test to assign constant material properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
        - mass_rho: Density [kg/m³]
        - epsilonTh: Thomsen parameter epsilon
        - gammaTh: Thomsen parameter gamma
        - deltaTh: Thomsen parameter delta
        - thetaTTI: Tilt angle in degrees
        - phiTTI: Azimuth angle in degrees (phi = 0: 2D case)
    '''

    Wave_obj = wave_instance

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Uniform initial distribution
    vel_P_o = 1500
    vel_S_o = 750
    mass_rho_o = 1e3
    epsilonTh_o = 0.2
    gammaTh_o = 0.3
    deltaTh_o = 0.1
    thetaTTI_o = 30.
    phiTTI_o = 15.
    constant_lst = [vel_P_o, vel_S_o, mass_rho_o, epsilonTh_o,
                    gammaTh_o, deltaTh_o, thetaTTI_o, phiTTI_o]

    print("\nTesting Constant Material Properties", flush=True)
    for prop_name, constant in zip(scalar_mat_prop, constant_lst):
        try:
            mat_property = Wave_obj.set_material_property(
                prop_name, 'scalar', constant=constant,
                output=True, foldername='/property_fields/constant/')

            assert mat_property is not None, f"Failed to set {prop_name}"

            # Get the mean value function to verify
            dx = fire.dx(**Wave_obj.quadrature_rule)
            dummy_vol = Wave_obj.set_material_property('dummy_vol',
                                                       'scalar',
                                                       constant=1.)
            volume = fire.assemble(dummy_vol * dx)
            mean_val = fire.assemble(mat_property * dx) / volume

            assert np.isclose(mean_val, constant, rtol=1e-8), \
                f"❌ {prop_name}: Expected value {constant}, got {mean_val}"
            print(f"✅ {prop_name} Verified: expected "
                  f"{constant}, got = {round(mean_val, 10)}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Setting {prop_name} raised an exception: {str(e)}")


def test_random_mat_prop(wave_instance):
    '''
    Test to assign random material properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
        - mass_rho: Density [kg/m³]
        - epsilonTh: Thomsen parameter epsilon
        - gammaTh: Thomsen parameter gamma
        - deltaTh: Thomsen parameter delta
        - thetaTTI: Tilt angle in degrees
        - phiTTI: Azimuth angle in degrees (phi = 0: 2D case)
    '''

    Wave_obj = wave_instance

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Random initial distribution
    random_lst = [(1.5e3, 2e3), (750, 1e3), (1e3, 2e3), (0.1, 0.3),
                  (0.2, 0.4), (-0.1, 0.2), (-60, 60), (-15, 15)]

    print("\nTesting Random Material Properties", flush=True)
    for prop_name, random in zip(scalar_mat_prop, random_lst):
        try:
            mat_property = Wave_obj.set_material_property(
                prop_name, 'scalar', random=random,
                output=True, foldername='/property_fields/random/')

            assert mat_property is not None, f"Failed to set {prop_name}"

            # Verify values are within range
            mat_property_data = mat_property.dat.data_with_halos
            min_val = random[0]
            max_val = random[1]

            assert np.all(mat_property_data >= min_val - 1e-8), \
                f"❌ Values below minimum {min_val} for {prop_name}"
            print(f"✅ {prop_name} Verified: Values "
                  f">= minimum {min_val}", flush=True)
            assert np.all(mat_property_data <= max_val + 1e-8), \
                f"❌ Values above maximum {max_val} for {prop_name}"
            print(f"✅ {prop_name} Verified: Values "
                  f"<= maximum {max_val}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Setting {prop_name} raised an exception: {str(e)}")


def numerical_values_cond(prop_name, coords, below_thrs, above_thrs):
    '''
    Compute the expected numerical values for the conditional material
    property based on the provided coordinates and property name.

    Parameters
    ----------
    property_name: `str`
            Name of the material property to be se
    coords : `numpy.ndarray`
        Array of coordinates (z, x, y) for the mesh points
    below_thrs : `numpy.ndarray`
        Boolean array indicating points below the threshold condition
    above_thrs : `numpy.ndarray`
        Boolean array indicating points above the threshold condition

    Returns
    -------
    exp_below : `numpy.ndarray`
        Expected values for points below the threshold condition
    exp_above : `numpy.ndarray`
        Expected values for points above the threshold condition
    '''
    z = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]

    if prop_name == 'vel_P':
        exp_below = 2. + abs(z[below_thrs])
        exp_above = 1.5
    elif prop_name == 'vel_S':
        exp_below = (2. + abs(z[below_thrs])) / 2.5
        exp_above = 0.75
    elif prop_name == 'mass_rho':
        exp_below = 1.7e3 + 3e3 * abs(z[below_thrs]) ** 2
        exp_above = 1e3
    elif prop_name == 'epsilonTh':
        exp_below = np.exp(x[below_thrs]) / 10.
        exp_above = 0.15 * np.exp(x[above_thrs])
    elif prop_name == 'gammaTh':
        exp_below = 2.5 * np.exp(x[below_thrs]) / 10.
        exp_above = np.exp(x[above_thrs]) / 10.
    elif prop_name == 'deltaTh':
        exp_below = -np.exp(x[below_thrs]) / 20.
        exp_above = np.exp(x[above_thrs]) / 10.
    elif prop_name == 'thetaTTI':
        exp_below = 1e4 * (y[below_thrs] - 0.08)**2 / 2. - 2.
        exp_above = -(1e4 * (y[above_thrs] - 0.08)**2 / 2. - 2.)
    elif prop_name == 'phiTTI':
        exp_below = -1e4 * (y[below_thrs] - 0.08)**2 / 2. + 2.
        exp_above = -(-1e4 * (y[above_thrs] - 0.08)**2 / 2. + 2.)

    return exp_below, exp_above


def get_only_mesh_vertices(Wave_obj):
    '''
    Get the coordinates of the mesh vertices and
    the indices for the points in the mesh function.

    Parameters
    ----------
    Wave_obj : `wave.IsotropicWave`
        An instance of the IsotropicWave class

    Returns
    -------
    coords : `numpy.ndarray`
        Array of coordinates (z, x, y) for the mesh vertices
    mask_pnt : `numpy.ndarray`
        Array of indices in the function corresponding to the mesh vertices
    '''

    mesh_f = fire.Function(
        Wave_obj.function_space).interpolate(Wave_obj.mesh.coordinates)
    coords = Wave_obj.mesh.coordinates.dat.data_with_halos
    mask_pnt = np.where(np.isin(
        mesh_f.dat.data_with_halos, coords).all(axis=1))[0]

    return coords, mask_pnt


def test_conditional_mat_prop(wave_instance):
    '''
    Test to assign conditional material properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity[m/s]
        - vel_S: S-wave velocity[m/s]
        - mass_rho: Density[kg/m³]
        - epsilonTh: Thomsen parameter epsilon
        - gammaTh: Thomsen parameter gamma
        - deltaTh: Thomsen parameter delta
        - thetaTTI: Tilt angle in degrees
        - phiTTI: Azimuth angle in degrees(phi=0: 2D case)
    '''

    Wave_obj = wave_instance

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Conditional initial distribution
    f_vel = 2. + abs(Wave_obj.mesh_z)
    cond_Vp = fire.conditional(Wave_obj.mesh_z < -0.06, f_vel, 1.5)
    cond_Vs = fire.conditional(Wave_obj.mesh_z < -0.06, f_vel / 2.5, 0.75)
    f_rho = 1.7e3 + 3e3 * abs(Wave_obj.mesh_z) ** 2
    cond_rho = fire.conditional(Wave_obj.mesh_z < -0.06, f_rho, 1e3)
    f_TH = fire.exp(Wave_obj.mesh_x) / 10.
    cond_eps = fire.conditional(Wave_obj.mesh_x < 0.28, f_TH, 1.5 * f_TH)
    cond_gam = fire.conditional(Wave_obj.mesh_x < 0.28, 2.5 * f_TH, f_TH)
    cond_del = fire.conditional(Wave_obj.mesh_x < 0.28, -f_TH / 2., f_TH)
    f_TTI = 1e4 * (Wave_obj.mesh_y - 0.08)**2 / 2. - 2.
    cond_the = fire.conditional(Wave_obj.mesh_y < 0.08, f_TTI, -f_TTI)
    cond_phi = fire.conditional(Wave_obj.mesh_y < 0.08, -f_TTI, f_TTI)
    cond_lst = [cond_Vp, cond_Vs, cond_rho, cond_eps,
                cond_gam, cond_del, cond_the, cond_phi]

    # Threshold values for testing
    threshold_dict = {'vel_P': -0.06,
                      'vel_S': -0.06,
                      'mass_rho': -0.06,
                      'epsilonTh': 0.28,
                      'gammaTh': 0.28,
                      'deltaTh': 0.28,
                      'thetaTTI': 0.08,
                      'phiTTI': 0.08}

    # Get mesh vertices
    coords, mask_pnt = get_only_mesh_vertices(Wave_obj)

    print("\nTesting Conditional Material Properties", flush=True)
    for prop_name, cond_field in zip(scalar_mat_prop, cond_lst):
        try:
            mat_property = Wave_obj.set_material_property(
                prop_name, 'scalar', conditional=cond_field,
                output=True, foldername='/property_fields/conditional/')

            assert mat_property is not None, f"Failed to set {prop_name}"

            threshold = threshold_dict[prop_name]

            if prop_name in ['vel_P', 'vel_S', 'mass_rho']:
                below_thrs = coords[:, 0] < threshold
                above_thrs = coords[:, 0] >= threshold
                coord_thrs = 'z'

            if prop_name in ['epsilonTh', 'gammaTh', 'deltaTh']:
                below_thrs = coords[:, 1] < threshold
                above_thrs = coords[:, 1] >= threshold
                coord_thrs = 'x'

            if prop_name in ['thetaTTI', 'phiTTI']:
                below_thrs = coords[:, 2] < threshold
                above_thrs = coords[:, 2] >= threshold
                coord_thrs = 'y'

            exp_below, exp_above = numerical_values_cond(
                prop_name, coords, below_thrs, above_thrs)
            cnd_below = mat_property.dat.data_with_halos[mask_pnt][below_thrs]
            cnd_above = mat_property.dat.data_with_halos[mask_pnt][above_thrs]

            assert np.allclose(exp_below, cnd_below, rtol=1e-8), \
                f"❌ Values does not match for {prop_name}" + \
                f" for {coord_thrs} < {threshold}"
            print(f"✅ {prop_name} Verified: Conditional values "
                  f"{coord_thrs} < {threshold}", flush=True)
            assert np.allclose(exp_above, cnd_above, rtol=1e-8), \
                f"❌ Values does not match for {prop_name}" + \
                f" for {coord_thrs} >= {threshold}"
            print(f"✅ {prop_name} Verified: Conditional values "
                  f"{coord_thrs} >= {threshold}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Setting {prop_name} raised an exception: {str(e)}")


def numerical_values_expr(prop_name, coords):
    '''
    Compute the expected numerical values for the expression material
    property based on the provided coordinates and property name.

    Parameters
    ----------
    property_name: `str`
            Name of the material property to be se
    coords : `numpy.ndarray`
        Array of coordinates (z, x, y) for the mesh vertices

    Returns
    -------
    exp_num : `numpy.ndarray`
        Expected values for the expression material property
    '''
    z = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]

    if prop_name == 'vel_P':
        exp_num = 1.5e3 * (1 + np.sqrt(x**2 + y**2 + z**2))
    elif prop_name == 'vel_S':
        exp_num = 1e3 * (0.7 + np.sqrt(x**2 + y**2 + z**2))
    elif prop_name == 'mass_rho':
        exp_num = 1e3 * (1 + 5 * np.log(1 + x**2 + y**2 + z**2))
    elif prop_name == 'epsilonTh':
        exp_num = np.sin(x) * np.cos(y) / 4 + np.sin(y) * np.cos(z) / 3 + 0.1
    elif prop_name == 'gammaTh':
        exp_num = np.sin(y) * np.cos(z) / 4 + np.sin(z) * np.cos(x) / 3 + 0.3
    elif prop_name == 'deltaTh':
        exp_num = np.sin(z) * np.cos(x) / 2 + np.sin(x) * np.cos(y) / 3 + 0.02
    elif prop_name == 'thetaTTI':
        exp_num = np.atan2(np.abs(z), x) * 180 / np.pi - 45
    elif prop_name == 'phiTTI':
        d = 1e-16  # Small constant to avoid division by zero
        f = 180 / np.pi
        exp_num = f * np.acos(y / (np.sqrt(x**2 + y**2 + z**2) + d)) - 45

    return exp_num


def test_expression_mat_prop(wave_instance):
    '''
    Test to assign expressions as material properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity[m/s]
        - vel_S: S-wave velocity[m/s]
        - mass_rho: Density[kg/m³]
        - epsilonTh: Thomsen parameter epsilon
        - gammaTh: Thomsen parameter gamma
        - deltaTh: Thomsen parameter delta
        - thetaTTI: Tilt angle in degrees
        - phiTTI: Azimuth angle in degrees(phi=0: 2D case)
    '''

    Wave_obj = wave_instance

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',  #
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Expression for initial distribution
    expr_lst = ["1.5e3 * (1 + sqrt(x**2 + y**2 + z**2))",
                "1e3 * (0.7 + sqrt(x**2 + y**2 + z**2))",
                "1e3 * (1 + 5 * ln(1 + x**2 + y**2 + z**2))",
                "sin(x)*cos(y) / 4 + sin(y)*cos(z) / 3 + 0.1",
                "sin(y)*cos(z) / 4 + sin(z)*cos(x) / 3 + 0.3",
                "sin(z)*cos(x) / 2 + sin(x)*cos(y) / 3 + 0.02",
                "atan2(sqrt(z**2), x) * 180 / pi - 45",
                "180 / pi * acos(y / (sqrt(x**2 + y**2 + z**2) + 1e-16)) - 45"]

    # Get mesh vertices
    coords, mask_pnt = get_only_mesh_vertices(Wave_obj)

    print("\nTesting Expression as Material Properties", flush=True)
    for prop_name, expr_field in zip(scalar_mat_prop, expr_lst):
        try:
            mat_property = Wave_obj.set_material_property(
                prop_name, 'scalar', expression=expr_field,
                output=True, foldername='/property_fields/expression/')

            assert mat_property is not None, f"Failed to set {prop_name}"

            exp_exp = numerical_values_expr(prop_name, coords)
            val_exp = mat_property.dat.data_with_halos[mask_pnt]

            assert np.allclose(val_exp, exp_exp, rtol=1e-8), \
                f"❌ Values of the expression does not match for {prop_name}"
            print(f"✅ {prop_name} Verified: Expression values", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Setting {prop_name} raised an exception: {str(e)}")


def get_coords_DG0(Wave_obj, coords):
    '''
    Compute the coordinates of the cell centroids for DG0 interpolation.

    Parameters
    ----------
    Wave_obj : `wave.IsotropicWave`
        An instance of the IsotropicWave class
    coords : `numpy.ndarray`
        Array of coordinates (z, x, y) for the mesh vertices

    Returns
    -------
    coords_DG0 : `numpy.ndarray`
        Array of coordinates (z, x, y) for the cell centroids
        for DG0 interpolation
    '''

    coords_DG0 = coords[:]
    coords_DG0[:, 0] -= Wave_obj.mesh_parameters.edge_length / 2.
    coords_DG0[:, 1] += Wave_obj.mesh_parameters.edge_length / 2.
    coords_DG0[:, 2] += Wave_obj.mesh_parameters.edge_length / 2.
    coords_DG0.round(2)

    return coords_DG0


def test_function_mat_prop(wave_instance):
    '''
    Test to assign firedrake functione as material
    properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
    '''

    Wave_obj = wave_instance

    print("\nTesting Firedrake Functions as Material Properties", flush=True)
    dummy_expr = "7.5e2 * (1 + sqrt(x**2 + y**2 + z**2))"
    dummy = Wave_obj.set_material_property(
        'dummy', 'scalar', expression=dummy_expr, output=False)

    # Same function space
    vel_S = Wave_obj.set_material_property(
        'vel_S', 'scalar', fire_function=dummy, output=True,
        foldername='/property_fields/function/')

    assert vel_S is not None, "Failed to set vel_S"

    # Get mesh vertices from original mesh
    coords, mask_pnt = get_only_mesh_vertices(Wave_obj)

    exp_fun = dummy.dat.data_with_halos[mask_pnt]
    val_fun = vel_S.dat.data_with_halos[mask_pnt]

    assert np.allclose(val_fun, exp_fun, rtol=1e-8), \
        "❌ Values of the firedrake function does not match for vel_S"
    print("✅ vel_S Verified: Firedrake function values", flush=True)

    # Different function space (DG0)
    vel_S_dg0 = Wave_obj.set_material_property(
        'vel_S_DG0', 'scalar', fire_function=vel_S, dg_property=True,
        output=True, foldername='/property_fields/function/')

    assert vel_S_dg0 is not None, "Failed to set vel_S_DG0"

    # Coordinates of the cell centroids for DG0 interpolation
    coords_DG0 = get_coords_DG0(Wave_obj, coords)

    exp_fun = dummy.at(coords_DG0, dont_raise=True)
    val_fun = vel_S.at(coords_DG0, dont_raise=True)

    assert np.allclose(val_fun, exp_fun, rtol=1e-8), \
        "❌ Values of the scalar function does not match for vel_S_DG0"
    print("✅ vel_S_DG0 Verified: Scalar function values", flush=True)


def test_fromfile_mat_prop(wave_instance):
    '''
    Test to assign firedrake functione as material
    properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
    '''

    Wave_obj = wave_instance

    print("\nTesting File Inputs as Material Properties", flush=True)
    vel_P = Wave_obj.set_material_property(
        'vel_P', 'scalar', constant=1., output=True,
        foldername='/property_fields/from_file/')

    dummy = Wave_obj.set_material_property('dummy', 'scalar', constant=1.)
    dummy.dat.data_with_halos[:] = vel_P.dat.data_with_halos[:] / 2.

    from_file_segy = getcwd() + '/property_fields/from_file/vel_S.segy'
    create_segy(dummy, Wave_obj.function_space.sub(0),
                Wave_obj.mesh_parameters.edge_length, from_file_segy)

    with pytest.raises(ValueError) as exc_info:
        vel_S = Wave_obj.set_material_property(   # noqa: F841
            'vel_S', 'scalar', from_file=from_file_segy, output=True,
            foldername='/property_fields/from_file/')

    # Verify the error message
    expected_message = "Error Setting a Material Property: " + \
        "Initializing property from file is currently not implemented"
    assert expected_message in str(exc_info.value), \
        f"❌ Unexpected error message: {str(exc_info.value)}"

    print(" ✅ Material Property from file: Correctly raised "
          f"NotImplementedError: {exc_info.value}", flush=True)


def test_vector_mat_prop(wave_instance):
    '''
    Test to assign vector material properties to an instance of Wave.

    Material properties:
        - alphaT: Thermal expansion vector [ppm/°C]
    '''

    Wave_obj = wave_instance
    print("\nTesting Vector Material Properties", flush=True)
    print("Vector: Thermal Expansion Field", flush=True)

    # Same function space
    alphaT_o = 1e-5
    print("\nTesting Constant Material Properties", flush=True)
    alphaT_cte = Wave_obj.set_material_property(
        "alphaT", 'vector', constant=alphaT_o, output=True,
        foldername='/property_fields/vector_tensor/')

    assert alphaT_cte is not None, "Failed to set alphaT_cte"

    dummy = Wave_obj.set_material_property('dummy', 'vector', constant=0.)
    dummy.sub(0).assign(Wave_obj.set_material_property('dummy_z', 'scalar',
                                                       constant=alphaT_o))
    dummy.sub(1).assign(Wave_obj.set_material_property('dummy_x', 'scalar',
                                                       constant=2*alphaT_o))
    dummy.sub(2).assign(Wave_obj.set_material_property('dummy_y', 'scalar',
                                                       constant=3*alphaT_o))

    # Different function space (DG0)
    alphaT_dg0 = Wave_obj.set_material_property(
        'alphaT_DG0', 'vector', fire_function=dummy, dg_property=True,
        output=True, foldername='/property_fields/vector_tensor/')

    assert alphaT_dg0 is not None, "Failed to set alphaT_dg0"

    # Get the mean value vectorial component to verify
    dx = fire.dx(**Wave_obj.quadrature_rule)
    dummy_vol = Wave_obj.set_material_property('dummy_vol',
                                               'scalar',
                                               constant=1.)
    volume = fire.assemble(dummy_vol * dx)

    exp_v0 = fire.assemble(dummy.sub(0) * dx) / volume
    val_v0 = fire.assemble(alphaT_dg0.sub(0) * dx) / volume
    exp_v1 = fire.assemble(dummy.sub(1) * dx) / volume
    val_v1 = fire.assemble(alphaT_dg0.sub(1) * dx) / volume
    exp_v2 = fire.assemble(dummy.sub(2) * dx) / volume
    val_v2 = fire.assemble(alphaT_dg0.sub(2) * dx) / volume
    cond0 = np.isclose(exp_v0, val_v0, rtol=1e-8)
    cond1 = np.isclose(exp_v1, val_v1, rtol=1e-8)
    cond2 = np.isclose(exp_v2, val_v2, rtol=1e-8)

    assert cond0 and cond1 and cond2, \
        "❌ Values of the vectorial function does not match for alphaT_dg0"
    print("✅ alphaT_dg0 Verified: Vectorial function values", flush=True)


def test_tensor_mat_prop(wave_instance):
    '''
    Test to assign tensor material properties to an instance of Wave.

    Material properties:
        - C: Elastic Tensor [GPa]
    '''

    Wave_obj = wave_instance
    print("\nTesting Tensor Material Properties", flush=True)
    print("Vector: Elastic Anisotropic Tensor", flush=True)

    C11 = 3.15
    C33 = 2.25
    C44 = 0.5625
    C66 = 0.9
    C12 = 1.35
    C13 = 1.4656

    C_elast = fire.as_tensor(((C11, C12, C13, 0, 0, 0),
                              (C12, C11, C13, 0, 0, 0),
                              (C13, C13, C33, 0, 0, 0),
                              (0, 0, 0, C44, 0, 0),
                              (0, 0, 0, 0, C44, 0),
                              (0, 0, 0, 0, 0, C66)))

    shape_func_space = C_elast.ufl_shape

    # Tensor 6x6
    dummy = Wave_obj.set_material_property('dummy', 'tensor',
                                           shape_func_space=shape_func_space,
                                           constant=0.)

    entries = []
    for i in range(shape_func_space[0]):
        row = []
        for j in range(shape_func_space[1]):
            val = float(C_elast[i, j])
            if val != 0.0:
                # Only create a scalar property if the entry is nonzero
                row.append(Wave_obj.set_material_property(
                    f'dummy_{i+1}{j+1}', 'scalar', constant=val))
            else:
                # Keep an explicit zero so the tensor shape stays consistent
                row.append(0.0)
        entries.append(row)
    tensor_expr = fire.as_tensor(entries)

    # Interpolate into dummy
    dummy.interpolate(tensor_expr)

    # Same function space
    Celast = Wave_obj.set_material_property(
        'Celast', 'tensor', fire_function=dummy,
        shape_func_space=shape_func_space)

    assert Celast is not None, "Failed to set Celast"
    print("✅ Celast Verified: Tensorial function assign", flush=True)

    # Tensor 2x3
    dummy = Wave_obj.set_material_property('dummy', 'tensor',
                                           shape_func_space=(2, 3),
                                           constant=0.)

    compC_name = ['C11', 'C33', 'C44', 'C66', 'C12', 'C13']
    compC_val = fire.as_tensor(((C11, C33, C44),
                                (C66, C12, C13)))

    entries = []
    for i in range(2):
        row = []
        for j in range(3):
            val = float(compC_val[i, j])
            row.append(Wave_obj.set_material_property(
                compC_name[3 * i + j], 'scalar',
                constant=val))
        entries.append(row)
    tensor_expr = fire.as_tensor(entries)

    # Interpolate into dummy
    dummy.interpolate(tensor_expr)

    # Same function space
    Celast_2x3 = Wave_obj.set_material_property(
        'Celast_2x3', 'tensor', shape_func_space=(2, 3),
        fire_function=dummy, output=True,
        foldername='/property_fields/vector_tensor/')

    assert Celast_2x3 is not None, "Failed to set Celast_2x3"
    print("✅ Celast_2x3 Verified: Tensorial function saving", flush=True)

    # Different function space (DG0)
    Celast_dg0 = Wave_obj.set_material_property(
        'Celast_DG0', 'tensor', shape_func_space=(2, 3),
        fire_function=dummy, dg_property=True, output=True,
        foldername='/property_fields/vector_tensor/')

    assert Celast_dg0 is not None, "Failed to set Celast_dg0"
    print("✅ Celast_2x3 Verified: Tensorial function saving DG0", flush=True)
