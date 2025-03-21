# --- Basid setup to run a forward simulation with AD --- #
import firedrake as fire
import spyro


def model_settings():
    """Model settings for forward and Full Waveform Inversion (FWI)
    simulations.

    Returns
    -------
    model : dict
        Dictionary containing the model settings.
    """

    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or mass_lumped_triangle
        "quadrature": "KMV",  # Equi or mass_lumped_triangle
        "degree": 1,  # p order
        "dimension": 2,  # dimension
        "regularization": False,  # regularization is on?
        "gamma": 1e-5,  # regularization parameter
    }

    model["parallelism"] = {
        # options:
        # `shots_parallelism`. Shots parallelism.
        # None - no shots parallelism.
        "type": "shots_parallelism",
        "num_spacial_cores": 1,  # Number of cores to use in the spatial
        # parallelism.
    }

    # Define the domain size without the ABL.
    model["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "not_used.msh",
        "initmodel": "not_used.hdf5",
        "truemodel": "not_used.hdf5",
    }

    # Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
    model["BCs"] = {
        "status": False,  # True or False, used to turn on any type of BC
        "outer_bc": "non-reflective",  # none or non-reflective (outer boundary condition)
        "abl_bc": "none",  # none, gaussian-taper, or alid
        "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
        "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "source_pos": spyro.create_transect((0.2, 0.15), (0.8, 0.15), 3),
        "frequency": 7.0,
        "delay": 1.0,
        "receiver_locations": spyro.create_transect((0.2, 0.2), (0.8, 0.2), 10),
    }
    model["aut_dif"] = {
        "status": True,
        "checkpointing": True,
    }

    model["timeaxis"] = {
        "t0": 0.0,  # Initial time for event
        "tf": 0.8,  # Final time for event (for test 7)
        "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 20,  # (20 for dt=0.00050) how frequently to output solution to pvds
        "fspool": 1,  # how frequently to save solution to RAM
    }

    return model


def make_c_camembert(mesh, function_space, c_guess=False, plot_c=False):
    """Acoustic velocity model.

    Parameters
    ----------
    mesh : firedrake.Mesh
        Mesh.
    function_space : firedrake.FunctionSpace
        Function space.
    c_guess : bool, optional
        If True, the initial guess for the velocity field is returned.
    plot_c : bool, optional
        If True, the velocity field is saved to a VTK file.
    """
    x, z = fire.SpatialCoordinate(mesh)
    if c_guess:
        c = fire.Function(function_space).interpolate(1.5 + 0.0 * x)
    else:
        c = fire.Function(function_space).interpolate(
            2.5
            + 1 * fire.tanh(100 * (0.125 - fire.sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    if plot_c:
        outfile = fire.VTKFile("acoustic_cp.pvd")
        outfile.write(c)
    return c
