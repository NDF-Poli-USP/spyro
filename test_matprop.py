import firedrake as fire
import warnings
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(domain_dim, tf_usu, dt_usu, fr_files):
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
    fr_files : `int`
        Frequency of the output files to be saved in the simulation

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
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
        "source_type": "ricker",
        "source_locations": [(-Lz / 2., Lx / 2., Ly / 2.)],
        "frequency": 5.0,  # in Hz
        "delay": 1. / 3.,
        "delay_type": "time",  # "multiples_of_minimum" or "time"
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                               (0., 0., 0), (0., Lx, 0.),
                               (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1.5 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": tf_usu,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
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
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho',
                       'epsilonTh', 'gammaTh', 'deltaTh']

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
        Wave_obj.set_material_property(
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
    scalar_mat_prop = ['vel_P', 'vel_S', 'mass_rho',
                       'epsilonTh', 'gammaTh', 'deltaTh']

    # Random initial distribution
    random_lst = [(1.5e3, 2e3), (750, 1e3), (1e3, 2e3), (0.1, 0.3),
                  (0.2, 0.4), (-0.1, 0.2), (-60, 60), (-15, 15)]

    print("\nTesting Random Material Properties", flush=True)
    for property_name, random in zip(scalar_mat_prop, random_lst):
        Wave_obj.set_material_property(
            property_name, 'scalar', random=random,
            output=True, foldername='/property_fields/random/')


def instance_wave():
    '''
    Create an instance of the isotropic wave solver.

    ave_obj : `wave.IsotropicWave`
        An instance of the IsotropicWave class
    '''

    # Domain dimensions
    domain_dim = [0.24, 0.56, 0.16]  # in km

    # Final Time
    tf_usu = 2.  # s

    # Number of timesteps
    steps = 200

    # Frequency of output files
    fr_files = 1

    # Time step of the simulation
    dt_usu = round(tf_usu / steps, 6)

    # Mesh size in km
    edge_length = 0.040

    # Create dictionary with parameters for the model
    dictionary = wave_dict(domain_dim, tf_usu, dt_usu, fr_files)

    # Create a wave object
    Wave_obj = IsotropicWave(dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    return Wave_obj


# Testing anisotropy solver with NRBC and explosive source in 3D
if __name__ == "__main__":
    test_constant_mat_prop()
    test_random_mat_prop()
