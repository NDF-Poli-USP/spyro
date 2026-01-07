import firedrake as fire
import warnings
from spyro.solvers.elastic_wave import AnisotropicWave
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_elast_dict(domain_dim, tf_usu, dt_usu, fr_files):
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
        "source_type": "moment",
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

    # Define Parameters for absorving boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "nrbc",  # Activate NRBCs
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output/forward/fw_output_3d.pvd",
        "mechanical_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output/preamble/acoustic_pot_energy_3d",
    }

    dictionary["material"] = {
        "type": "VTI",  # Anisotropic material type
        "type_parameters": "constant",  # constant or range (provide min/max)
        "density": (1.5e3, 2e3),  # in kg/m^3
    }

    return dictionary


def test_prop_elast_ani_3d():
    '''
    Testing function for anisotropic elastic wave
    propagation in 3D with NRBC and explosive source.
    '''

    domain_dim, tf_usu, dt_usu, fr_files = basic_parameters()

    dictionary = wave_elast_dict(domain_dim, tf_usu, dt_usu, fr_files)

    wave_obj = spyro.AnisotropicWave(dictionary)
    # wave_obj.set_mesh(user_mesh=mesh, input_mesh_parameters={})


def basic_parameters():

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

    return domain_dim, tf_usu, dt_usu, fr_files


# Testing anisotropy solver with NRBC and explosive source in 3D
if __name__ == "__main__":
    test_prop_elast_ani_3d()
