from firedrake import *  # noqa: F403
from FIAT.reference_element import (
    UFCTriangle,
    UFCTetrahedron,
    UFCQuadrilateral,
)
from FIAT.reference_element import UFCInterval
from FIAT import GaussLobattoLegendre as GLLelement
from FIAT.tensor_product import TensorProductElement
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen as KMV
from FIAT.lagrange import Lagrange as CG
from FIAT.discontinuous_lagrange import DiscontinuousLagrange as DG

import numpy as np


class Receivers:
    """Interpolate data defined on a triangular mesh to a
    set of 2D/3D coordinates for variable spatial order
    using Lagrange interpolation.

    Can interpolate receiveir values that do not coincide with
    mesh or DOF points

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
    receiver_locations: list
        List of tuples containing all receiver locations
    num_receivers: int
        Number of receivers
    quadrilateral: boolean
        Boolean that specifies if cells are quadrilateral
    is_local: list of booleans
        List that checks if receivers are present in cores
        spatial paralelism

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_receivers_as_source(rhs_forcing, residual, IT)
        Applies receivers as source with values from residual
        in timestep IT, for usage with adjoint propagation
    """

    def __init__(self, wave_object):
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
        self.point_locations = wave_object.receiver_locations

        if self.dimension == 3 and wave_object.automatic_adjoint:
            # self.column_x = model["acquisition"]["num_rec_x_columns"]
            # self.column_y = model["acquisition"]["num_rec_y_columns"]
            # self.column_z = model["acquisition"]["num_rec_z_columns"]
            # self.number_of_points = self.column_x*self.column_y
            raise ValueError("Implement this later")
        else:
            self.number_of_points = wave_object.number_of_receivers

        self.cellIDs = None
        self.cellVertices = None
        self.cell_tabulations = None
        self.cellNodeMaps = None
        self.nodes_per_cell = None
        if wave_object.cell_type == "quadrilateral":
            self.quadrilateral = True
        else:
            self.quadrilateral = False
        self.is_local = [0] * self.number_of_points
        if not self.automatic_adjoint:
            self.build_maps()

    @property
    def num_receivers(self):
        return self.__num_receivers

    @num_receivers.setter
    def num_receivers(self, value):
        if value <= 0:
            raise ValueError("No receivers specified")
        self.__num_receivers = value

    def build_maps(self):
        """Calculates and stores tabulations for interpolation

        Is always automatticaly called when initializing the class,
        therefore should only be called again if a mesh related attribute
        changes.

        Returns
        -------
        cellIDs: list
            List of cell IDs for each receiver
        cellVertices: list
            List of vertices for each receiver
        cellNodeMaps: list
            List of node maps for each receiver
        cell_tabulations: list
            List of tabulations for each receiver
        """

        for rid in range(self.number_of_points):
            tolerance = 1e-6
            if self.dimension == 2:
                receiver_z, receiver_x = self.point_locations[rid]
                cell_id = self.mesh.locate_cell(
                    [receiver_z, receiver_x], tolerance=tolerance
                )
            elif self.dimension == 3:
                receiver_z, receiver_x, receiver_y = self.point_locations[
                    rid
                ]
                cell_id = self.mesh.locate_cell(
                    [receiver_z, receiver_x, receiver_y], tolerance=tolerance
                )
            self.is_local[rid] = cell_id

        (
            self.cellIDs,
            self.cellVertices,
            self.cellNodeMaps,
        ) = self.__func_receiver_locator()
        self.cell_tabulations = self.__func_build_cell_tabulations()

        self.number_of_points = len(self.point_locations)

    def interpolate(self, field):
        """Interpolate the solution to the receiver coordinates for
        one simulation timestep.

        Parameters
        ----------
        field: array-like
            An array of the solution at a given timestep at all nodes

        Returns
        -------
        solution_at_receivers: list
            Solution interpolated to the list of receiver coordinates
            for the given timestep.
        """
        return [self.__new_at(field, rn) for rn in range(self.number_of_points)]

    def apply_receivers_as_source(self, rhs_forcing, residual, IT):
        """The adjoint operation of interpolation (injection)

        Injects residual, and timestep IT, at receiver locations
        as source and stores their value in the right hand side
        operator rhs_forcing.

        Parameters
        ----------
        rhs_forcing: object
            Firedrake assembled right hand side operator to store values
        residual: list
            List of residual values at different receiver locations
            and timesteps
        IT: int
            Desired time step number to get residual value from

        Returns
        -------
        rhs_forcing: object
            Firedrake assembled right hand side operator with injected values
        """
        for rid in range(self.number_of_points):
            value = residual[IT][rid]
            if self.is_local[rid]:
                idx = np.int_(self.cellNodeMaps[rid])
                phis = self.cell_tabulations[rid]

                tmp = np.dot(phis, value)
                rhs_forcing.dat.data_with_halos[idx] += tmp
            else:
                tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing

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
        num_recv = self.number_of_points

        fdrake_cell_node_map = self.space.cell_node_map()
        cell_node_map = fdrake_cell_node_map.values_with_halo
        (num_cells, nodes_per_cell) = cell_node_map.shape
        node_locations = self.__func_node_locations()
        self.nodes_per_cell = nodes_per_cell

        cellId_maps = np.zeros((num_recv, 1))
        cellNodeMaps = np.zeros((num_recv, nodes_per_cell))
        cellVertices = []

        if self.quadrilateral is True:
            end_vertex_id = 4
            degree = self.degree
            cell_ends = [
                0,
                (degree + 1) * (degree + 1) - degree - 1,
                (degree + 1) * (degree + 1) - 1,
                degree,
            ]
        else:
            end_vertex_id = 3
            cell_ends = [0, 1, 2]

        for receiver_id in range(num_recv):
            cell_id = self.is_local[receiver_id]

            cellVertices.append([])

            if cell_id is not None:
                cellId_maps[receiver_id] = cell_id
                cellNodeMaps[receiver_id, :] = cell_node_map[cell_id, :]
                for vertex_number in range(0, end_vertex_id):
                    cellVertices[receiver_id].append([])
                    z = node_locations[
                        cell_node_map[cell_id, cell_ends[vertex_number]], 0
                    ]
                    x = node_locations[
                        cell_node_map[cell_id, cell_ends[vertex_number]], 1
                    ]
                    cellVertices[receiver_id][vertex_number] = (z, x)

        return cellId_maps, cellVertices, cellNodeMaps

    def __new_at(self, udat, receiver_id):
        """Function that evaluates the receiver value given its id.
        For 2D simplices only.
        Parameters
        ----------
        udat: array-like
            An array of the solution at a given timestep at all nodes
        receiver_id: a list of integers
            A list of receiver ids, ranging from 0 to total receivers
            minus one.

        Returns
        -------
        at: Function value at given receiver
        """

        if self.is_local is not None:
            # Getting relevant receiver points
            u = udat[np.int_(self.cellNodeMaps[receiver_id, :])]
        else:
            return udat[0]  # junk receiver isn't local

        phis = self.cell_tabulations[receiver_id, :]

        at = phis.T @ u

        return at

    def __func_node_locations_2D(self):
        """Function that returns a list which includes a numpy matrix
        where line n has the x and y values of the nth degree of freedom,
        and a numpy matrix of the vertex coordinates.
        """
        z, x = SpatialCoordinate(self.mesh)  # noqa: F405
        ux = Function(self.space).interpolate(x)  # noqa: F405
        uz = Function(self.space).interpolate(z)  # noqa: F405
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
        print("start func_receiver_locator", flush=True)
        num_recv = self.number_of_points

        fdrake_cell_node_map = self.space.cell_node_map()
        cell_node_map = fdrake_cell_node_map.values_with_halo
        if self.quadrilateral is True:
            cell_node_map = get_hexa_real_cell_node_map(self.space, self.mesh)
        (num_cells, nodes_per_cell) = cell_node_map.shape
        node_locations = self.__func_node_locations()
        self.nodes_per_cell = nodes_per_cell

        cellId_maps = np.zeros((num_recv, 1))
        cellNodeMaps = np.zeros((num_recv, nodes_per_cell))
        cellVertices = []

        if self.quadrilateral is True:
            end_vertex = 8
            p = self.degree
            vertex_ids = [
                0,
                p,
                (p + 1) * p,
                (p + 1) * p + p,
                (p + 1) * (p + 1) * p,
                (p + 1) * (p + 1) * p + p,
                (p + 1) * (p + 1) * p + (p + 1) * p,
                (p + 1) ** 3 - 1,
            ]
        else:
            end_vertex = 4
            vertex_ids = [0, 1, 2, 3]

        for receiver_id in range(num_recv):
            cell_id = self.is_local[receiver_id]
            cellVertices.append([])
            if cell_id is not None:
                cellId_maps[receiver_id] = cell_id
                cellNodeMaps[receiver_id, :] = cell_node_map[cell_id, :]
                for vertex_number in range(0, end_vertex):
                    vertex_id = vertex_ids[vertex_number]
                    cellVertices[receiver_id].append([])
                    z = node_locations[cell_node_map[cell_id, vertex_id], 0]
                    x = node_locations[cell_node_map[cell_id, vertex_id], 1]
                    y = node_locations[cell_node_map[cell_id, vertex_id], 2]
                    cellVertices[receiver_id][vertex_number] = (z, x, y)

        print("end func_receiver_locator", flush=True)
        return cellId_maps, cellVertices, cellNodeMaps

    def __func_node_locations_3D(self):
        """Function that returns a list which includes a numpy matrix
        where line n has the x and y values of the nth degree of freedom,
        and a numpy matrix of the vertex coordinates.

        """
        x, y, z = SpatialCoordinate(self.mesh)  # noqa: F405
        ux = Function(self.space).interpolate(x)  # noqa: F405
        uy = Function(self.space).interpolate(y)  # noqa: F405
        uz = Function(self.space).interpolate(z)  # noqa: F405
        datax = ux.dat.data_ro_with_halos[:]
        datay = uy.dat.data_ro_with_halos[:]
        dataz = uz.dat.data_ro_with_halos[:]
        node_locations = np.zeros((len(datax), 3))
        node_locations[:, 0] = datax
        node_locations[:, 1] = datay
        node_locations[:, 2] = dataz
        return node_locations

    def __func_build_cell_tabulations(self):
        if self.dimension == 2 and self.quadrilateral is False:
            return self.__func_build_cell_tabulations_2D()
        elif self.dimension == 3 and self.quadrilateral is False:
            return self.__func_build_cell_tabulations_3D()
        elif self.dimension == 2 and self.quadrilateral is True:
            return self.__func_build_cell_tabulations_2D_quad()
        elif self.dimension == 3 and self.quadrilateral is True:
            return self.__func_build_cell_tabulations_3D_quad()
        else:
            raise ValueError

    def __func_build_cell_tabulations_2D(self):
        element = choosing_element(self.space, self.degree)

        cell_tabulations = np.zeros((self.number_of_points, self.nodes_per_cell))

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
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

        cell_tabulations = np.zeros((self.number_of_points, self.nodes_per_cell))

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]

                p_reference = change_to_reference_tetrahedron(
                    p, v0, v1, v2, v3
                )
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def __func_build_cell_tabulations_2D_quad(self):
        # finatelement = FiniteElement('CG', self.mesh.ufl_cell(),
        # degree=self.degree, variant='spectral')
        V = self.space

        element = V.finat_element.fiat_equivalent

        cell_tabulations = np.zeros((self.number_of_points, self.nodes_per_cell))

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]

                p_reference = change_to_reference_quad(p, v0, v1, v2, v3)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def __func_build_cell_tabulations_3D_quad(self):
        Inter = UFCInterval()
        An = GLLelement(Inter, self.degree)
        Bn = GLLelement(Inter, self.degree)
        Cn = GLLelement(Inter, self.degree)
        Dn = TensorProductElement(An, Bn)
        element = TensorProductElement(Dn, Cn)

        cell_tabulations = np.zeros((self.number_of_points, self.nodes_per_cell))

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]
                v4 = self.cellVertices[receiver_id][4]
                v5 = self.cellVertices[receiver_id][5]
                v6 = self.cellVertices[receiver_id][6]
                v7 = self.cellVertices[receiver_id][7]

                p_reference = change_to_reference_hexa(
                    p, v0, v1, v2, v3, v4, v5, v6, v7
                )
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def set_point_cloud(self, comm):
        # Receivers always parallel to z-axis

        rec_pos = self.point_locations

        # 2D --
        if self.dimension == 2:
            num_rec = self.number_of_points
            δz = np.linspace(rec_pos[0, 0], rec_pos[num_rec - 1, 0], 1)
            δx = np.linspace(rec_pos[0, 1], rec_pos[num_rec - 1, 1], num_rec)

            Z, X = np.meshgrid(δz, δx)
            xs = np.vstack((Z.flatten(), X.flatten())).T

        # 3D
        elif self.dimension == 3:
            δz = np.linspace(rec_pos[0][0], rec_pos[1][0], self.column_z)
            δx = np.linspace(rec_pos[0][1], rec_pos[1][1], self.column_x)
            δy = np.linspace(rec_pos[0][2], rec_pos[1][2], self.column_y)

            Z, X, Y = np.meshgrid(δz, δx, δy)
            xs = np.vstack((Z.flatten(), X.flatten(), Y.flatten())).T
        else:
            print("This dimension is not accepted.")
            quit()

        point_cloud = VertexOnlyMesh(self.mesh, xs)  # noqa: F405

        return point_cloud


# Some helper functions


def choosing_element(V, degree):
    """Chooses UFL element based on desired function space
    and degree of interpolation.

    Parameters
    ----------
    V : firedrake.FunctionSpace
        Function space to be used.
    degree : int
        Degree of interpolation.

    Returns
    -------
    element : UFL element
        UFL element to be used in the interpolation.
    """
    cell_geometry = V.mesh().ufl_cell()
    if cell_geometry == quadrilateral:  # noqa: F405
        T = UFCQuadrilateral()
        raise ValueError(
            "Point interpolation for quads implemented somewhere else."
        )

    elif cell_geometry == triangle:  # noqa: F405
        T = UFCTriangle()

    elif cell_geometry == tetrahedron:  # noqa: F405
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
    """Changes variables to reference triangle"""
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
    a11 = (
        -(xnb * ya - xnc * ya - xna * yb + xnc * yb + xna * yc - xnb * yc)
        / div
    )
    a12 = (
        xa * xnb - xa * xnc - xb * xna + xb * xnc + xc * xna - xc * xnb
    ) / div
    a13 = (
        xa * xnc * yb
        - xb * xnc * ya
        - xa * xnb * yc
        + xc * xnb * ya
        + xb * xna * yc
        - xc * xna * yb
    ) / div
    a21 = (
        -(ya * ynb - ya * ync - yb * yna + yb * ync + yc * yna - yc * ynb)
        / div
    )
    a22 = (
        xa * ynb - xa * ync - xb * yna + xb * ync + xc * yna - xc * ynb
    ) / div
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
    """Changes variables to reference tetrahedron"""
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
        (xnc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (xnd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (xnb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (xna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a12 = (
        (xnd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (xnc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (xnb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (xna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a13 = (
        (xnc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (xnd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (xnb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (xna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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
        (ync * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (ynd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (ynb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (yna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a22 = (
        (ynd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (ync * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (ynb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (yna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a23 = (
        (ync * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (ynd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (ynb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (yna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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
        (znc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (znd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (znb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (zna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a32 = (
        (znd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (znc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (znb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (zna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a33 = (
        (znc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (znd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (znb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (zna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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


def change_to_reference_quad(p, v0, v1, v2, v3):
    """Changes varibales to reference quadrilateral"""
    (px, py) = p
    # Irregular quad
    (x0, y0) = v0
    (x1, y1) = v1
    (x2, y2) = v2
    (x3, y3) = v3

    # Reference quad
    # xn0 = 0.0
    # yn0 = 0.0
    # xn1 = 1.0
    # yn1 = 0.0
    # xn2 = 1.0
    # yn2 = 1.0
    # xn3 = 0.0
    # yn3 = 1.0

    dx1 = x1 - x2
    dx2 = x3 - x2
    dy1 = y1 - y2
    dy2 = y3 - y2
    sumx = x0 - x1 + x2 - x3
    sumy = y0 - y1 + y2 - y3

    gover = np.array([[sumx, dx2], [sumy, dy2]])

    g_under = np.array([[dx1, dx2], [dy1, dy2]])

    gunder = np.linalg.det(g_under)

    hover = np.array([[dx1, sumx], [dy1, sumy]])

    hunder = gunder

    g = np.linalg.det(gover) / gunder
    h = np.linalg.det(hover) / hunder
    i = 1.0

    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    c = x0
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    f = y0

    A = e * i - f * h
    B = c * h - b * i
    C = b * f - c * e
    D = f * g - d * i
    E = a * i - c * g
    F = c * d - a * f
    G = d * h - e * g
    H = b * g - a * h
    Ij = a * e - b * d

    pnx = (A * px + B * py + C) / (G * px + H * py + Ij)
    pny = (D * px + E * py + F) / (G * px + H * py + Ij)

    return (pnx, pny)


def change_to_reference_hexa(p, v0, v1, v2, v3, v4, v5, v6, v7):
    (px, py, pz) = p
    # Irregular hexa
    a = v0
    b = v1
    c = v2
    d = v4

    (xa, ya, za) = a
    (xb, yb, zb) = b
    (xc, yc, zc) = c
    (xd, yd, zd) = d
    (px, py, pz) = p

    xna = 0.0
    yna = 0.0
    zna = 0.0

    xnb = 0.0
    ynb = 0.0
    znb = 1.0

    xnc = 0.0
    ync = 1.0
    znc = 0.0

    xnd = 1.0
    ynd = 0.0
    znd = 0.0

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
        (xnc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (xnd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (xnb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (xna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a12 = (
        (xnd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (xnc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (xnb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (xna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a13 = (
        (xnc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (xnd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (xnb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (xna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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
        (ync * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (ynd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (ynb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (yna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a22 = (
        (ynd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (ync * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (ynb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (yna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a23 = (
        (ync * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (ynd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (ynb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (yna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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
        (znc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (znd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (znb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (zna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a32 = (
        (znd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (znc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (znb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (zna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a33 = (
        (znc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (znd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (znb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (zna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
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


def get_hexa_real_cell_node_map(V, mesh):
    weird_cnm_func = V.cell_node_map()
    weird_cnm = weird_cnm_func.values_with_halo
    cells_per_layer, nodes_per_cell = np.shape(weird_cnm)
    layers = mesh.cell_set.layers - 1
    ufl_element = V.ufl_element()
    _, p = ufl_element.degree()

    cell_node_map = np.zeros(
        (layers * cells_per_layer, nodes_per_cell), dtype=int
    )
    print(f"cnm size : {np.shape(cell_node_map)}", flush=True)

    for layer in range(layers):
        print(f"layer : {layer} of {layers}", flush=True)
        for cell in range(cells_per_layer):
            cnm_base = weird_cnm[cell]
            cell_id = layer + layers * cell
            cell_node_map[cell_id] = [item + layer * (p) for item in cnm_base]

    return cell_node_map
