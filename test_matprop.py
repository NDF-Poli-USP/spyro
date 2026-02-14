import firedrake as fire
import warnings
from os import getcwd
from spyro.io.basicio import create_segy
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
from spyro.utils.eval_functions_to_ufl import generate_ufl_functions
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


def test_constant_mat_prop():
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

    Wave_obj = instance_wave()

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
    for property_name, constant in zip(scalar_mat_prop, constant_lst):
        mat_property = Wave_obj.set_material_property(
            property_name, 'scalar', constant=constant,
            output=True, foldername='/property_fields/constant/')


def test_random_mat_prop():
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

    Wave_obj = instance_wave()

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Random initial distribution
    random_lst = [(1.5e3, 2e3), (750, 1e3), (1e3, 2e3), (0.1, 0.3),
                  (0.2, 0.4), (-0.1, 0.2), (-60, 60), (-15, 15)]

    print("\nTesting Random Material Properties", flush=True)
    for property_name, random in zip(scalar_mat_prop, random_lst):
        mat_property = Wave_obj.set_material_property(
            property_name, 'scalar', random=random,
            output=True, foldername='/property_fields/random/')


def test_conditional_mat_prop():
    '''
    Test to assign conditional material properties to an instance of Wave.

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

    Wave_obj = instance_wave()

    # Material properties for testing
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho', 'epsilonTh',
                       'gammaTh', 'deltaTh', 'thetaTTI', 'phiTTI']

    # Conditional initial distribution
    f_vel = 2. + abs(Wave_obj.mesh_z)
    cond_Vp = fire.conditional(Wave_obj.mesh_z < -0.06, f_vel, 1.5)
    cond_Vs = fire.conditional(Wave_obj.mesh_z < -0.06, f_vel / 2.5, 0.75)
    f_rho = 1.7e3 + 3e3 * abs(Wave_obj.mesh_z ** 2)
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

    print("\nTesting Conditional Material Properties", flush=True)
    for property_name, cond_field in zip(scalar_mat_prop, cond_lst):
        mat_property = Wave_obj.set_material_property(
            property_name, 'scalar', conditional=cond_field,
            output=True, foldername='/property_fields/conditional/')


def test_expression_mat_prop():
    '''
    Test to assign expressions as material properties to an instance of Wave.

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

    Wave_obj = instance_wave()

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

    print("\nTesting Expression as Material Properties", flush=True)
    for property_name, expr_field in zip(scalar_mat_prop, expr_lst):
        mat_property = Wave_obj.set_material_property(
            property_name, 'scalar', expression=expr_field,
            output=True, foldername='/property_fields/expression/')


def test_function_mat_prop():
    '''
    Test to assign firedrake functione as material
    properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
    '''

    Wave_obj = instance_wave()

    print("\nTesting Firedrake Functions as Material Properties", flush=True)
    dummy_expr = "7.5e2 * (1 + sqrt(x**2 + y**2 + z**2))"
    dummy = Wave_obj.set_material_property(
        'dummy', 'scalar', expression=dummy_expr, output=False)

    # Same function space
    vel_S = Wave_obj.set_material_property(
        'vel_S', 'scalar', fire_function=dummy, output=True,
        foldername='/property_fields/function/')

    # Different function space (DG0)
    vel_S_dg0 = Wave_obj.set_material_property(
        'vel_S_DG0', 'scalar', fire_function=vel_S, dg_property=True,
        output=True, foldername='/property_fields/function/')


def test_fromfile_mat_prop():
    '''
    Test to assign firedrake functione as material
    properties to an instance of Wave.

    Material properties:
        - vel_P: P-wave velocity [m/s]
        - vel_S: S-wave velocity [m/s]
    '''

    Wave_obj = instance_wave()

    print("\nTesting File Inputs as Material Properties", flush=True)
    vel_P = Wave_obj.set_material_property(
        'vel_P', 'scalar', constant=1., output=True,
        foldername='/property_fields/from_file/')

    dummy = Wave_obj.set_material_property('dummy', 'scalar', constant=1.)
    dummy.dat.data_with_halos[:] = vel_P.dat.data_with_halos[:] / 2.

    from_file_segy = getcwd() + '/property_fields/from_file/vel_S.segy'
    create_segy(dummy, Wave_obj.function_space.sub(0),
                Wave_obj.mesh_parameters.edge_length, from_file_segy)
    vel_S = Wave_obj.set_material_property(
        'vel_S', 'scalar', from_file=from_file_segy, output=True,
        foldername='/property_fields/from_file/')


def test_vector_mat_prop():
    '''
    Test to assign vector material properties to an instance of Wave.

    Material properties:
        - alphaT: Thermal expansion vector [ppm/°C]
    '''

    Wave_obj = instance_wave()
    print("\nTesting Vector Material Properties", flush=True)
    print("Vector: Thermal Expansion Field", flush=True)

    # Same function space
    alphaT_o = 1.7e-5
    print("\nTesting Constant Material Properties", flush=True)
    alphaT_cte = Wave_obj.set_material_property(
        "alphaT", 'vector', constant=alphaT_o, output=True,
        foldername='/property_fields/vector_tensor/')

    dummy = Wave_obj.set_material_property('dummy', 'vector', constant=0.)
    dummy.sub(0).assign(Wave_obj.set_material_property('dummy_z', 'scalar',
                                                       constant=alphaT_o))
    dummy.sub(1).assign(Wave_obj.set_material_property('dummy_x', 'scalar',
                                                       random=(1e-5, 2e-5)))
    cond_z = fire.conditional(Wave_obj.mesh_y < 0.08, 1e-5, 1.7e-5)
    dummy.sub(2).assign(Wave_obj.set_material_property('dummy_y', 'scalar',
                                                       conditional=cond_z))

    # Different function space (DG0)
    alphaT_dg0 = Wave_obj.set_material_property(
        'alphaT_DG0', 'vector', fire_function=dummy, dg_property=True,
        output=True, foldername='/property_fields/vector_tensor/')


def test_tensor_mat_prop():
    '''
    Test to assign tensor material properties to an instance of Wave.

    Material properties:
        - C: Elastic Tensor [GPa]
    '''

    Wave_obj = instance_wave()
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
    Celast = Wave_obj.set_material_property(
        'Celast', 'tensor', shape_func_space=(2, 3),
        fire_function=dummy, output=True,
        foldername='/property_fields/vector_tensor/')

    # Different function space (DG0)
    Celast_dg0 = Wave_obj.set_material_property(
        'Celast_DG0', 'tensor', shape_func_space=(2, 3),
        fire_function=dummy, dg_property=True, output=True,
        foldername='/property_fields/vector_tensor/')


def instance_wave():
    '''
    Create an instance of the isotropic wave solver.

    ave_obj : `wave.IsotropicWave`
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


# Testing anisotropy solver with NRBC and explosive source in 3D
if __name__ == "__main__":
    # test_constant_mat_prop()
    # test_random_mat_prop()
    # test_conditional_mat_prop()
    # test_expression_mat_prop()
    # test_function_mat_prop()
    # test_fromfile_mat_prop()
    # test_vector_mat_prop()
    test_tensor_mat_prop()
