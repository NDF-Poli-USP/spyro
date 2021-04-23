from firedrake import *
from FIAT.reference_element import UFCTriangle, UFCTetrahedron
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen as KMV
from FIAT.lagrange import Lagrange as CG
from FIAT.discontinuous_lagrange import DiscontinuousLagrange as DG

import numpy as np


class Receivers:
    """Interpolate data defined on a triangular mesh to a
    set of 2D/3D coordinates for variable spatial order
    using Lagrange interpolation.
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
        Receivers: :class: 'Receiver' object

        """

        self.mesh = mesh
        self.space = V
        self.my_ensemble = my_ensemble
        self.dimension = model["opts"]["dimension"]
        self.degree = model["opts"]["degree"]

        self.num_receivers = model["acquisition"]["num_receivers"]
        self.receiver_locations = model["acquisition"]["receiver_locations"]

        self.cellIDs = None
        self.cellVertices = None
        self.cell_tabulations = None
        self.cellNodeMaps = None
        self.nodes_per_cell = None

    @property
    def num_receivers(self):
        return self.__num_receivers

    @num_receivers.setter
    def num_receivers(self, value):
        if value <= 0:
            raise ValueError("No receivers specified")
        self.__num_receivers = value

    def create(self):
        """Initialzies maps used in point interpolation"""

        (
            self.cellIDs,
            self.cellVertices,
            self.cellNodeMaps,
        ) = self.__func_receiver_locator()
        self.cell_tabulations = self.__func_build_cell_tabulations()
        # __build_local_nodes()

        self.num_receivers = len(self.receiver_locations)

        return self

    def interpolate(self, field, is_local):
        """Interpolate the solution to the receiver coordinates for
        one simulation timestep.

        Parameters
        ----------
        field: array-like
            An array of the solution at a given timestep at all nodes
        is_local: a list of booleans
            A list of integers. Positive if the receiver is local to
            the subdomain and negative otherwise.

        Returns
        -------
        solution_at_receivers: list
            Solution interpolated to the list of receiver coordinates
            for the given timestep.

        """
        return [
            self.__new_at(field, rn, is_local[rn]) for rn in range(self.num_receivers)
        ]

    def __func_receiver_locator(self):
        """Function that returns a list of tuples and a matrix
        the list of tuples has in line n the receiver position
        and the position of the nodes in the element that contains
        the receiver.

        The matrix has the deegres of freedom of the nodes inside
        same element as the receiver.

        """
        if self.dimension == 2:
            return self.__func_receiver_locator_2D()
        elif self.dimension == 3:
            return self.__func_receiver_locator_3D()
        else:
            raise ValueError

    def __build_local_nodes(self):
        """Builds local element nodes, locations and I,J,K numbering"""
        if self.dimension == 2:
            return self.__build_local_nodes_2D()
        elif self.dimension == 3:
            return self.__build_local_nodes_3D()
        else:
            raise ValueError

    def __func_node_locations(self):
        """Function that returns a list which includes a numpy matrix
        where line n has the x and y values of the nth degree of freedom,
        and a numpy matrix of the vertex coordinates.
        """
        if self.dimension == 2:
            return self.__func_node_locations_2D()
        elif self.dimension == 3:
            return self.__func_node_locations_3D()
        else:
            raise ValueError

    def __func_receiver_locator_2D(self):
        """Function that returns a list of tuples and a matrix
        the list of tuples has in line n the receiver position
        and the position of the nodes in the element that contains
        the receiver.

        The matrix has the deegres of freedom of the nodes inside
        same element as the receiver.

        """
        num_recv = self.num_receivers

        fdrake_cell_node_map = self.space.cell_node_map()
        cell_node_map = fdrake_cell_node_map.values_with_halo
        (num_cells, nodes_per_cell) = cell_node_map.shape
        node_locations = self.__func_node_locations()
        self.nodes_per_cell = nodes_per_cell

        cellId_maps = np.zeros((num_recv, 1))
        cellNodeMaps = np.zeros((num_recv, nodes_per_cell))
        cellVertices = []

        for receiver_id in range(num_recv):
            (receiver_z, receiver_x) = self.receiver_locations[receiver_id]

            cell_id = self.mesh.locate_cell([receiver_z, receiver_x], tolerance=0.0100)

            cellVertices.append([])

            if cell_id is not None:
                cellId_maps[receiver_id] = cell_id
                cellNodeMaps[receiver_id, :] = cell_node_map[cell_id, :]
                for vertex_number in range(0, 3):
                    cellVertices[receiver_id].append([])
                    z = node_locations[cell_node_map[cell_id, vertex_number], 0]
                    x = node_locations[cell_node_map[cell_id, vertex_number], 1]
                    cellVertices[receiver_id][vertex_number] = (z, x)

        return cellId_maps, cellVertices, cellNodeMaps

    def __new_at(self, udat, receiver_id, is_local):
        """Function that evaluates the receiver value given its id.
        For 2D simplices only.

        Parameters
        ----------
        udat: array-like
            An array of the solution at a given timestep at all nodes
        receiver_id: a list of integers
            A list of receiver ids, ranging from 0 to total receivers
            minus one.
        is_local: a list of booleans
            A list of integers. Positive if the receiver is local to
            the subdomain and negative otherwise.

        Returns
        -------

        at: Function value at given receiver
        """

        if is_local is not None:
            # Getting relevant receiver points
            u = udat[np.int_(self.cellNodeMaps[receiver_id, :])]
        else:
            return udat[0]  # junk receiver isn't local

        phis = self.cell_tabulations[receiver_id, :]
        at = 0

        for i in range(len(u)):

            at += phis[i] * u[i]

        return float(at)

    def __func_node_locations_2D(self):
        """Function that returns a list which includes a numpy matrix
        where line n has the x and y values of the nth degree of freedom,
        and a numpy matrix of the vertex coordinates.
        """
        z, x = SpatialCoordinate(self.mesh)
        ux = Function(self.space).interpolate(x)
        uz = Function(self.space).interpolate(z)
        datax = ux.dat.data_ro_with_halos[:]
        dataz = uz.dat.data_ro_with_halos[:]
        node_locations = np.zeros((len(datax), 2))
        node_locations[:, 0] = dataz
        node_locations[:, 1] = datax

        return node_locations

    def __func_receiver_locator_3D(self):
        """Function that returns a list of tuples and a matrix
        the list of tuples has in line n the receiver position
        and the position of the nodes in the element that contains
        the receiver.

        The matrix has the deegres of freedom of the nodes inside
        same element as the receiver.

        """
        num_recv = self.num_receivers

        fdrake_cell_node_map = self.space.cell_node_map()
        cell_node_map = fdrake_cell_node_map.values_with_halo
        (num_cells, nodes_per_cell) = cell_node_map.shape
        node_locations = self.__func_node_locations()
        self.nodes_per_cell = nodes_per_cell

        cellId_maps = np.zeros((num_recv, 1))
        cellNodeMaps = np.zeros((num_recv, nodes_per_cell))
        cellVertices = []

        for receiver_id in range(num_recv):
            (receiver_z, receiver_x, receiver_y) = self.receiver_locations[receiver_id]

            cell_id = self.mesh.locate_cell(
                [receiver_z, receiver_x, receiver_y], tolerance=0.0100
            )
            cellVertices.append([])
            if cell_id is not None:
                cellId_maps[receiver_id] = cell_id
                cellNodeMaps[receiver_id, :] = cell_node_map[cell_id, :]
                for vertex_number in range(0, 4):
                    cellVertices[receiver_id].append([])
                    z = node_locations[cell_node_map[cell_id, vertex_number], 0]
                    x = node_locations[cell_node_map[cell_id, vertex_number], 1]
                    y = node_locations[cell_node_map[cell_id, vertex_number], 2]
                    cellVertices[receiver_id][vertex_number] = (z, x, y)

        return cellId_maps, cellVertices, cellNodeMaps

    def __func_node_locations_3D(self):
        """Function that returns a list which includes a numpy matrix
        where line n has the x and y values of the nth degree of freedom,
        and a numpy matrix of the vertex coordinates.
        """
        x, y, z = SpatialCoordinate(self.mesh)
        ux = Function(self.space).interpolate(x)
        uy = Function(self.space).interpolate(y)
        uz = Function(self.space).interpolate(z)
        datax = ux.dat.data_ro_with_halos[:]
        datay = uy.dat.data_ro_with_halos[:]
        dataz = uz.dat.data_ro_with_halos[:]
        node_locations = np.zeros((len(datax), 3))
        node_locations[:, 0] = datax
        node_locations[:, 1] = datay
        node_locations[:, 2] = dataz
        return node_locations

    def __func_build_cell_tabulations(self):
        if self.dimension == 2:
            return self.__func_build_cell_tabulations_2D()
        elif self.dimension == 3:
            return self.__func_build_cell_tabulations_3D()
        else:
            raise ValueError

    def __func_build_cell_tabulations_2D(self):

        element = choosing_element(self.space, self.degree)

        cell_tabulations = np.zeros((self.num_receivers, self.nodes_per_cell))

        for receiver_id in range(self.num_receivers):
            (receiver_z, receiver_x) = self.receiver_locations[receiver_id]
            cell_id = self.mesh.locate_cell([receiver_z, receiver_x], tolerance=0.0100)

            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.receiver_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]

                p_reference = change_to_reference_triangle(p, v0, v1, v2)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def __func_build_cell_tabulations_3D(self):
        element = choosing_element(self.space, self.degree)

        cell_tabulations = np.zeros((self.num_receivers, self.nodes_per_cell))

        for receiver_id in range(self.num_receivers):
            (receiver_z, receiver_x, receiver_y) = self.receiver_locations[receiver_id]
            cell_id = self.mesh.locate_cell(
                [receiver_z, receiver_x, receiver_y], tolerance=0.0100
            )

            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.receiver_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]

                p_reference = change_to_reference_tetrahedron(p, v0, v1, v2, v3)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations


## Some helper functions


def choosing_element(V, degree):
    cell_geometry = V.mesh().ufl_cell()
    if cell_geometry == quadrilateral:
        T = UFCQuadrilateral()
        raise ValueError("Point interpolation not yet implemented for quads")

    elif cell_geometry == triangle:
        T = UFCTriangle()

    elif cell_geometry == tetrahedron:
        T = UFCTetrahedron()

    else:
        raise ValueError("Unrecognized cell geometry.")

    if V.ufl_element().family() == "Kong-Mulder-Veldhuizen":
        element = KMV(T, degree)
    elif V.ufl_element().family() == "Lagrange":
        element = CG(T, degree)
    elif V.ufl_element().family() == "Discontinuous Lagrange":
        element = DG(T, degree)
    else:
        raise ValueError("Function space not yet supported.")

    return element


def change_to_reference_triangle(p, a, b, c):
    (xa, ya) = a
    (xb, yb) = b
    (xc, yc) = c
    (px, py) = p

    xna = 0.0
    yna = 0.0
    xnb = 1.0
    ynb = 0.0
    xnc = 0.0
    ync = 1.0

    div = xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb
    a11 = -(xnb * ya - xnc * ya - xna * yb + xnc * yb + xna * yc - xnb * yc) / div
    a12 = (xa * xnb - xa * xnc - xb * xna + xb * xnc + xc * xna - xc * xnb) / div
    a13 = (
        xa * xnc * yb
        - xb * xnc * ya
        - xa * xnb * yc
        + xc * xnb * ya
        + xb * xna * yc
        - xc * xna * yb
    ) / div
    a21 = -(ya * ynb - ya * ync - yb * yna + yb * ync + yc * yna - yc * ynb) / div
    a22 = (xa * ynb - xa * ync - xb * yna + xb * ync + xc * yna - xc * ynb) / div
    a23 = (
        xa * yb * ync
        - xb * ya * ync
        - xa * yc * ynb
        + xc * ya * ynb
        + xb * yc * yna
        - xc * yb * yna
    ) / div

    pnx = px * a11 + py * a12 + a13
    pny = px * a21 + py * a22 + a23

    return (pnx, pny)


def change_to_reference_tetrahedron(p, a, b, c, d):
    (xa, ya, za) = a
    (xb, yb, zb) = b
    (xc, yc, zc) = c
    (xd, yd, zd) = d
    (px, py, pz) = p

    xna = 0.0
    yna = 0.0
    zna = 0.0

    xnb = 1.0
    ynb = 0.0
    znb = 0.0

    xnc = 0.0
    ync = 1.0
    znc = 0.0

    xnd = 0.0
    ynd = 0.0
    znd = 1.0

    det = (
        xa * yb * zc
        - xa * yc * zb
        - xb * ya * zc
        + xb * yc * za
        + xc * ya * zb
        - xc * yb * za
        - xa * yb * zd
        + xa * yd * zb
        + xb * ya * zd
        - xb * yd * za
        - xd * ya * zb
        + xd * yb * za
        + xa * yc * zd
        - xa * yd * zc
        - xc * ya * zd
        + xc * yd * za
        + xd * ya * zc
        - xd * yc * za
        - xb * yc * zd
        + xb * yd * zc
        + xc * yb * zd
        - xc * yd * zb
        - xd * yb * zc
        + xd * yc * zb
    )
    a11 = (
        (xnc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb)) / det
        - (xnd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb)) / det
        - (xnb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc)) / det
        + (xna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc)) / det
    )
    a12 = (
        (xnd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb)) / det
        - (xnc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb)) / det
        + (xnb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc)) / det
        - (xna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc)) / det
    )
    a13 = (
        (xnc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb)) / det
        - (xnd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb)) / det
        - (xnb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc)) / det
        + (xna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc)) / det
    )
    a14 = (
        (
            xnd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            xnc
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            xnb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            xna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )
    a21 = (
        (ync * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb)) / det
        - (ynd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb)) / det
        - (ynb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc)) / det
        + (yna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc)) / det
    )
    a22 = (
        (ynd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb)) / det
        - (ync * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb)) / det
        + (ynb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc)) / det
        - (yna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc)) / det
    )
    a23 = (
        (ync * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb)) / det
        - (ynd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb)) / det
        - (ynb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc)) / det
        + (yna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc)) / det
    )
    a24 = (
        (
            ynd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            ync
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            ynb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            yna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )
    a31 = (
        (znc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb)) / det
        - (znd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb)) / det
        - (znb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc)) / det
        + (zna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc)) / det
    )
    a32 = (
        (znd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb)) / det
        - (znc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb)) / det
        + (znb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc)) / det
        - (zna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc)) / det
    )
    a33 = (
        (znc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb)) / det
        - (znd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb)) / det
        - (znb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc)) / det
        + (zna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc)) / det
    )
    a34 = (
        (
            znd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            znc
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            znb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            zna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )

    pnx = px * a11 + py * a12 + pz * a13 + a14
    pny = px * a21 + py * a22 + pz * a23 + a24
    pnz = px * a31 + py * a32 + pz * a33 + a34

    return (pnx, pny, pnz)
