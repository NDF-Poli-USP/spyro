from firedrake import File
import matplotlib.pyplot as plt
import numpy as np
import math
import spyro


def plot_receiver(
    receiver,
    receiver_id,
    dt,
    final_time,
    show=False,
    file_format="pdf",
):
    """Plot a

    Returns
    -------
    None
    """
    receiver_data = receiver[:, receiver_id]

    nt = int(final_time / dt)  # number of timesteps
    times = np.linspace(0.0, final_time, nt)

    plt.plot(times, receiver_data)

    plt.xlabel("time (s)", fontsize=18)
    plt.ylabel("amplitude", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(start_index, end_index)
    # plt.ylim(tf, 0)
    plt.savefig("receiver"+str(receiver_id) + "." + file_format, format=file_format)
    if show:
        plt.show()
    plt.close()
    return None


def compare_velocity(p_r, receiver_in_source_index, receiver_comparison_index, model, dt):
    receiver_0 = p_r[:, receiver_in_source_index]
    receiver_1 = p_r[:, receiver_comparison_index]
    pos = model["acquisition"]["receiver_locations"]
    time0 = np.argmax(receiver_0)*dt
    time1 = np.argmax(receiver_1)*dt
    x0 = pos[receiver_in_source_index, 1]
    x1 = pos[receiver_comparison_index, 1]
    measured_velocity = np.abs(x1-x0)/(time1-time0)
    minimum_velocity = 1.5
    error_percent = 100*np.abs(measured_velocity-minimum_velocity)/minimum_velocity
    print(f"Velocity error of {error_percent}%.", flush=True)
    return error_percent


def get_receiver_in_source_location(source_id, model):
    receiver_locations = model["acquisition"]["receiver_locations"]
    source_locations = model["acquisition"]["source_pos"]
    source_x = source_locations[source_id, 1]

    cont = 0
    for receiver_location in receiver_locations:
        if math.isclose(source_x, receiver_location[1]):
            return cont
        cont += 1
    return ValueError("Couldn't find a receiver whose location coincides with a source within the standard tolerance.")


def test_forward_5shots():
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV",  # Equi or KMV
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    model["parallelism"] = {
        "type": "automatic",
    }
    model["mesh"] = {
        "Lz": 3.5,  # depth in km - always positive
        "Lx": 17.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "meshes/marmousi_5Hz.msh",
        "initmodel": None,
        "truemodel": "velocity_models/vp_marmousi-ii.hdf5",
    }
    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
        "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
    }
    model["acquisition"] = {
        "source_type": "Ricker",
        "source_pos": spyro.create_transect((-0.1, 1.0), (-0.1, 15.0), 5),
        "frequency": 5.0,
        "delay": 1.0,
        "receiver_locations": spyro.create_transect((-0.1, 1.0), (-0.1, 15.0), 13),
    }
    model["timeaxis"] = {
        "t0": 0.0,  # Initial time for event
        "tf": 3.00,  # Final time for event
        "dt": 0.001,
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 100,  # how frequently to output solution to pvds
        "fspool": 99999,  # how frequently to save solution to RAM
    }

    dt = model["timeaxis"]["dt"]

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)
    vp = spyro.io.interpolate(model, mesh, V, guess=False)

    if comm.ensemble_comm.rank == 0:
        File("true_velocity.pvd", comm=comm.comm).write(vp)
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )
    p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)

    pass_error_test = False
    for source_id in range(len(model["acquisition"]["source_pos"])):
        if comm.ensemble_comm.rank == (source_id % comm.ensemble_comm.size):
            receiver_in_source_index = get_receiver_in_source_location(source_id, model)
            if source_id != len(model["acquisition"]["source_pos"])-1 or source_id == 0:
                receiver_comparison_index = receiver_in_source_index + 1
            else:
                receiver_comparison_index = receiver_in_source_index - 1
            error_percent = compare_velocity(p_r, receiver_in_source_index, receiver_comparison_index, model, dt)
            if error_percent < 5:
                pass_error_test = True
            print(f"For source = {source_id}: test = {pass_error_test}", flush=True)

    spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
    spyro.io.save_shots(model, comm, p_r)
    assert pass_error_test


if __name__ == "__main__":
    test_forward_5shots()
