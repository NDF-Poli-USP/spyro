import math
import numpy as np
from scipy.signal import butter, filtfilt
import spyro


class Sources(spyro.receivers.Receivers.Receivers):
    """Methods that inject a wavelet into a mesh

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V: Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble: Firedrake.ensemble_communicator
        An ensemble communicator
    dimension: int
        The dimension of the space
    degree: int
        Degree of the function space
    source_locations: list
        List of tuples containing all source locations
    num_sources: int
        Number of sources
    quadrilateral: boolean
        Boolean that specifies if cells are quadrilateral
    is_local: list of booleans
        List that checks if sources are present in cores
        spatial paralelism

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_source(rhs_forcing, value)
        Applies value at source locations in rhs_forcing operator
    """

    def __init__(self, model, mesh, V, my_ensemble):
        """Initializes class and gets all receiver parameters from
        input file.

        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        my_ensemble: Firedrake.ensemble_communicator
            An ensemble communicator

        Returns
        -------
        Sources: :class: 'Source' object

        """

        self.mesh = mesh
        self.space = V
        self.my_ensemble = my_ensemble
        self.dimension = model["opts"]["dimension"]
        self.degree = model["opts"]["degree"]

        self.receiver_locations = model["acquisition"]["source_pos"]
        self.num_receivers = len(self.receiver_locations)

        self.cellIDs = None
        self.cellVertices = None
        self.cell_tabulations = None
        self.cellNodeMaps = None
        self.nodes_per_cell = None
        self.is_local = [0]*self.num_receivers
        self.current_source = None
        self.quadrilateral = (model["opts"]['quadrature'] == 'GLL')

        super().build_maps()

    def apply_source(self, rhs_forcing, value):
        """Applies source in a assembled right hand side."""
        for source_id in range(self.num_receivers):
            if self.is_local[source_id] and source_id == self.current_source:
                for i in range(len(self.cellNodeMaps[source_id])):
                    rhs_forcing.dat.data_with_halos[int(self.cellNodeMaps[source_id][i])] = (
                        value * self.cell_tabulations[source_id][i]
                    )
            else:
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]  # noqa: F841

        return rhs_forcing


def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, amp, delay=delay)
    # elif model["acquisition"]["source_type"] == "MMS":
    #     return MMS_time(t)
    else:
        raise ValueError("source not implemented")


def ricker_wavelet(t, freq, amp=1.0, delay=1.5):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.
    """
    t = t - delay * math.sqrt(6.0) / (math.pi * freq)
    return (
        amp
        * (1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t)
        * math.exp(
            (-1.0 / 4.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )
    )


def full_ricker_wavelet(dt, tf, freq, amp=1.0, cutoff=None):
    """Compute the Ricker wavelet optionally applying low-pass filtering
    using cutoff frequency in Hertz.
    """
    nt = int(tf / dt)  # number of timesteps
    time = 0.0
    full_wavelet = np.zeros((nt,))
    for t in range(nt):
        full_wavelet[t] = ricker_wavelet(time, freq, amp)
        time += dt
    if cutoff is not None:
        fs = 1.0 / dt
        order = 2
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        full_wavelet = filtfilt(b, a, full_wavelet)
    return full_wavelet


# def MMS_time(t):
#     return 2 * t + 2 * math.pi ** 2 * t ** 3 / 3.0


# def MMS_space(x0, z, x):
#     """ Mesh variable part of the MMS """
#     return sin(pi * z) * sin(pi * x) * Constant(1.0)


# def MMS_space_3d(x0, z, x, y):
#     """ Mesh variable part of the MMS """
#     return sin(pi * z) * sin(pi * x) * sin(pi * y) * Constant(1.0)

# def source_dof_finder(space, model):

#     # getting 1 source position
#     source_positions = model["acquisition"]["source_pos"]
#     if len(source_positions) != 1:
#         raise ValueError("Not yet implemented for more then 1 source.")

#     mesh = space.mesh()
#     source_z, source_x = source_positions[0]

#     # Getting mesh coordinates
#     z, x = SpatialCoordinate(mesh)
#     ux = Function(space).interpolate(x)
#     uz = Function(space).interpolate(z)
#     datax = ux.dat.data_ro_with_halos[:]
#     dataz = uz.dat.data_ro_with_halos[:]
#     node_locations = np.zeros((len(datax), 2))
#     node_locations[:, 0] = dataz
#     node_locations[:, 1] = datax

#     # generating cell node map
#     fdrake_cell_node_map = space.cell_node_map()
#     cell_node_map = fdrake_cell_node_map.values_with_halo

#     # finding cell where the source is located
#     cell_id = mesh.locate_cell([source_z, source_x], tolerance=0.01)

#     # finding dof where the source is located
#     for dof in cell_node_map[cell_id]:
#         if np.isclose(dataz[dof], source_z, rtol=1e-8) and np.isclose(
#             datax[dof], source_x, rtol=1e-8
#         ):
#             model["acquisition"]["source_point_dof"] = dof

#     if model["acquisition"]["source_point_dof"] == False:
#         print("Warning not using point source")
#     return False


# def delta_expr(x0, z, x, sigma_x=500.0):
#     sigma_x = Constant(sigma_x)
#     return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))


# def delta_expr_3d(x0, z, x, y, sigma_x=2000.0):
#     sigma_x = Constant(sigma_x)
#     return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2 + (y - x0[2]) ** 2))
