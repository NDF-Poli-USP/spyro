import math
import numpy as np
from firedrake import *
from scipy.signal import butter, filtfilt
import spyro
from spyro.receivers.Receivers import *

class Sources(spyro.receivers.Receivers.Receivers):
    """Methods that inject a wavelet into a mesh"""

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
        Receivers: :class: 'Receiver' object

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
        self.cellNodes = None
        self.cell_tabulations = None
        self.cell_tabulations_xdir = None # tabulations for radial source, x direction
        self.cell_tabulations_zdir = None # tabulations for radial source, z direction
        self.cellNodeMaps = None
        self.nodes_per_cell = None
        self.node_locations = None
        self.is_local = [0]*self.num_receivers
        self.current_source = None

        super().build_maps()
        (self.cell_tabulations_zdir,self.cell_tabulations_xdir) = self.__func_build_cell_tabulations_zxdir()

    def apply_source(self, rhs_forcing, value):
        """Applies source in a assembled right hand side."""
        for source_id in range(self.num_receivers):
            if self.is_local[source_id] and source_id==self.current_source:
                for i in range(len(self.cellNodeMaps[source_id])):
                    rhs_forcing.dat.data_with_halos[int(self.cellNodeMaps[source_id][i])] = (
                        value * self.cell_tabulations[source_id][i]
                    )
            else: 
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing
    
    def apply_radial_source(self, rhs_forcing, value):
        """Applies radial sources in a assembled right hand side. Used in elastic waves"""
        # loop over the number of sources
        for source_id in range(self.num_receivers): # here, num_receivers means num_sources 
            if self.is_local[source_id] and source_id==self.current_source:
                #FIXME maybe use a flag to decide which one
                # loop over the nodes of the element that contains the source 
                #for i in range(len(self.cellNodeMaps[source_id])): 
                    # set the nodal values according to the wavelet (value) and the shape functions evaluated on each node
                #    rhs_forcing.sub(0).dat.data_with_halos[int(self.cellNodeMaps[source_id][i])] = (
                #        value * self.cell_tabulations_zdir[source_id][i]
                #    )
                #    rhs_forcing.sub(1).dat.data_with_halos[int(self.cellNodeMaps[source_id][i])] = (
                #        value * self.cell_tabulations_xdir[source_id][i]
                #    )
                rhs_forcing.sub(0).dat.data_with_halos[:] = (
                        value * self.cell_tabulations_zdir[source_id][:]
                )
                rhs_forcing.sub(1).dat.data_with_halos[:] = (
                        value * self.cell_tabulations_xdir[source_id][:]
                )
            
            else: 
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing

    def __func_build_cell_tabulations_zxdir(self):
        if self.dimension == 2:
            #return self.__func_build_cell_tabulations_zxdir_point_source() #FIXME maybe use a flag to decide which one
            return self.__func_build_cell_tabulations_zxdir_continuous_source()
        elif self.dimension == 3:
            raise ValueError("Point interpolation for 3D meshes not supported yet")
        else:
            raise ValueError
    
    def __func_build_cell_tabulations_zxdir_continuous_source(self):
        """Create tabulations over cells (actually nodes) considering
        a continuous source described by a Gaussian function.
        """
        num_nodes = self.node_locations.shape[0]
        cell_tabulations_xdir = np.zeros((self.num_receivers, num_nodes))
        cell_tabulations_zdir = np.zeros((self.num_receivers, num_nodes))

        nz = self.node_locations[:, 0]
        nx = self.node_locations[:, 1]

        for receiver_id in range(self.num_receivers): # actually source_id
            p = self.receiver_locations[receiver_id]
            (zdir, xdir) = directional_amplitudes_2D(p, nz, nx)
            cell_tabulations_xdir[receiver_id, :] = delta_expr2(p, nz, nx) * xdir
            cell_tabulations_zdir[receiver_id, :] = delta_expr2(p, nz, nx) * zdir

        return (cell_tabulations_zdir,cell_tabulations_xdir)
    

    def __func_build_cell_tabulations_zxdir_point_source(self):
        """Create tabulations over a cell containing the source position.
        Warning: this method should be used carefully in elastic waves
        simulations because the solution is mesh-dependent.
        """
        print("Warning: point source method called in elastic waves simulations")

        element = choosing_element(self.space, self.degree)

        cell_tabulations = np.zeros((self.num_receivers, self.nodes_per_cell))
        
        cell_tabulations_xdir = np.zeros((self.num_receivers, self.nodes_per_cell))
        cell_tabulations_zdir = np.zeros((self.num_receivers, self.nodes_per_cell))

        for receiver_id in range(self.num_receivers): # actually source_id
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.receiver_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]

                p_reference = change_to_reference_triangle(p, v0, v1, v2)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose() # values of the shape functions at the source pos.
                    
                # build the directional tabulation
                (pz, px) = p
                tol = 1e-6
                for node_id in range(self.nodes_per_cell):
                    (nz, nx) = self.cellNodes[receiver_id][node_id]
                    cell_tabulations_xdir[receiver_id,node_id] = (nx-px)/(tol + ((nx-px)**2.+(nz-pz)**2.)**0.5)
                    cell_tabulations_zdir[receiver_id,node_id] = (nz-pz)/(tol + ((nx-px)**2.+(nz-pz)**2.)**0.5)

        return (cell_tabulations*cell_tabulations_zdir,cell_tabulations*cell_tabulations_xdir)


def directional_amplitudes_2D(x0, z, x, tol=1e-6):
    # x0[0] -> z
    # x0[1] -> x
    r = ((x-x0[1])**2.+(z-x0[0])**2.)**0.5
    zdir = (z-x0[0])/(tol + r)
    xdir = (x-x0[1])/(tol + r)
    return (zdir,xdir)

def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, amp, delay=delay)
    elif model["acquisition"]["source_type"] == "MMS":
        return MMS_time(t)
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


def MMS_time(t):
    return 2 * t + 2 * math.pi ** 2 * t ** 3 / 3.0


def MMS_space(x0, z, x):
    """ Mesh variable part of the MMS """
    return sin(pi * z) * sin(pi * x) * Constant(1.0)


def MMS_space_3d(x0, z, x, y):
    """ Mesh variable part of the MMS """
    return sin(pi * z) * sin(pi * x) * sin(pi * y) * Constant(1.0)


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


def source_dof_finder(space, model):

    # getting 1 source position
    source_positions = model["acquisition"]["source_pos"]
    if len(source_positions) != 1:
        raise ValueError("Not yet implemented for more than 1 source.")

    mesh = space.mesh()
    source_z, source_x = source_positions[0]

    # Getting mesh coordinates
    z, x = SpatialCoordinate(mesh)
    ux = Function(space).interpolate(x)
    uz = Function(space).interpolate(z)
    datax = ux.dat.data_ro_with_halos[:]
    dataz = uz.dat.data_ro_with_halos[:]
    node_locations = np.zeros((len(datax), 2))
    node_locations[:, 0] = dataz
    node_locations[:, 1] = datax

    # generating cell node map
    fdrake_cell_node_map = space.cell_node_map()
    cell_node_map = fdrake_cell_node_map.values_with_halo

    # finding cell where the source is located
    cell_id = mesh.locate_cell([source_z, source_x], tolerance=0.01)

    # finding dof where the source is located
    for dof in cell_node_map[cell_id]:
        if np.isclose(dataz[dof], source_z, rtol=1e-8) and np.isclose(
            datax[dof], source_x, rtol=1e-8
        ):
            model["acquisition"]["source_point_dof"] = dof

    if model["acquisition"]["source_point_dof"] == False:
        print("Warning not using point source")
    return False


def delta_expr2(x0, z, x, sigma_x=500.0):
    return np.exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))


def delta_expr(x0, z, x, sigma_x=500.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))


def delta_expr_3d(x0, z, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2 + (y - x0[2]) ** 2))
