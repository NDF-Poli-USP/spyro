import numpy as np
from firedrake import *


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

        self.map1 = None
        self.map2 = None
        self.matrix_IJK = None

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

        self.map1, self.map2 = self.__func_receiver_locator()
        self.matrix_IJK = self.__build_local_nodes()

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

    def __new_at(self, udat, receiver_id, is_local):
        """Function that evaluates the receiver value given its id.

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
        at: function value at receiver locations
        """
        if self.dimension == 2:
            return self.__new_at_2D(udat, receiver_id, is_local)
        elif self.dimension == 3:
            return self.__new_at_3D(udat, receiver_id, is_local)
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

        receiver_maps_to_position = []
        receiver_maps_to_dof = np.zeros((num_recv, nodes_per_cell))

        for receiver_id in range(num_recv):
            (receiver_z, receiver_x) = self.receiver_locations[receiver_id]
            receiver_maps_to_position.append([])

            cell_id = self.mesh.locate_cell([receiver_z, receiver_x], tolerance=0.0100)

            if cell_id is not None:
                receiver_maps_to_position[receiver_id].append([])
                receiver_maps_to_position[receiver_id][0] = (receiver_z, receiver_x)
                for cont in range(nodes_per_cell):
                    z = node_locations[cell_node_map[cell_id, cont], 0]
                    x = node_locations[cell_node_map[cell_id, cont], 1]
                    receiver_maps_to_position[receiver_id].append((z, x))
                    receiver_maps_to_dof[receiver_id, cont] = cell_node_map[
                        cell_id, cont
                    ]
        return receiver_maps_to_position, receiver_maps_to_dof

    def __new_at_2D(self, udat, receiver_id, is_local):
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
            # Getting triangle/tetrahedron vertices and receiver point
            p = self.map1[receiver_id][0]
            v2 = self.map1[receiver_id][1]
            v1 = self.map1[receiver_id][2]
            v0 = self.map1[receiver_id][3]
            areaT = triangle_area(v0, v1, v2)

            u = udat[np.int_(self.map2[receiver_id, :])]
        else:
            return udat[0]  # junk receiver isn't local

        # Changing coordinates to L0, L1, L2 (area ratios)

        L0 = triangle_area(p, v1, v2) / areaT
        L1 = triangle_area(v0, p, v2) / areaT
        L2 = triangle_area(v0, v1, p) / areaT

        # Defining zeros for basis functions
        degree = self.degree

        zeros = []
        for i in range(degree + 1):
            zeros.append(i / degree)

        # summing over all basis functions
        at = 0

        for i in range(len(self.matrix_IJK)):
            I = self.matrix_IJK[i, 0]
            J = self.matrix_IJK[i, 1]
            K = self.matrix_IJK[i, 2]
            base1 = _lagrange_basis_1d(L0, I, I, zeros)
            base2 = _lagrange_basis_1d(L1, J, J, zeros)
            base3 = _lagrange_basis_1d(L2, K, K, zeros)
            unode = u[i]
            at += base1 * base2 * base3 * unode

        return at

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
        node_locations = np.zeros((len(datax), 3))
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

        receiver_maps_to_position = []
        receiver_maps_to_dof = np.zeros((num_recv, nodes_per_cell))

        for receiver_id in range(num_recv):
            (receiver_z, receiver_x, receiver_y) = self.receiver_locations[receiver_id]
            receiver_maps_to_position.append([])

            cell_id = self.mesh.locate_cell(
                [receiver_z, receiver_x, receiver_y], tolerance=0.0100
            )

            if cell_id is not None:
                receiver_maps_to_position[receiver_id].append([])
                receiver_maps_to_position[receiver_id][0] = (
                    receiver_z,
                    receiver_x,
                    receiver_y,
                )
                for cont in range(nodes_per_cell):
                    z = node_locations[cell_node_map[cell_id, cont], 0]
                    x = node_locations[cell_node_map[cell_id, cont], 1]
                    y = node_locations[cell_node_map[cell_id, cont], 2]
                    receiver_maps_to_position[receiver_id].append((z, x, y))
                    receiver_maps_to_dof[receiver_id, cont] = cell_node_map[
                        cell_id, cont
                    ]
        return receiver_maps_to_position, receiver_maps_to_dof

    def __new_at_3D(self, udat, receiver_id, is_local):
        """Function that evaluates the receiver value given its id.
        For 3D simplices only.

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

        Returns:
        -------
        at: function value at receveir location
        """

        if is_local is not None:
            # Getting triangle vertices and receiver point
            p = self.map1[receiver_id][0]
            v3 = self.map1[receiver_id][1]
            v2 = self.map1[receiver_id][3]
            v1 = self.map1[receiver_id][2]
            v0 = self.map1[receiver_id][4]
            volumeT = tetrahedral_volume(v0, v1, v2, v3)

            u = udat[np.int_(self.map2[receiver_id, :])]
        else:
            return udat[0]  # junk receiver isn't local

        # Changing coordinates to L0, L1, L2 (area ratios)
        L0 = tetrahedral_volume(p, v1, v2, v3) / volumeT
        L1 = tetrahedral_volume(v0, p, v2, v3) / volumeT
        L2 = tetrahedral_volume(v0, v1, p, v3) / volumeT
        L3 = tetrahedral_volume(v0, v1, v2, p) / volumeT

        # Defining zeros for basis functions
        degree = self.degree
        zeros = []
        for i in range(degree + 1):
            zeros.append(i / degree)

        # summing over all basis functions
        at = 0
        for i in range(len(self.matrix_IJK[0])):
            I = self.matrix_IJK[i, 0]
            J = self.matrix_IJK[i, 1]
            K = self.matrix_IJK[i, 2]
            Q = self.matrix_IJK[i, 3]
            base1 = _lagrange_basis_1d(L0, I, I, zeros)
            base2 = _lagrange_basis_1d(L1, J, J, zeros)
            base3 = _lagrange_basis_1d(L2, K, K, zeros)
            base4 = _lagrange_basis_1d(L3, Q, Q, zeros)
            unode = u[i]
            at += base1 * base2 * base3 * base4 * unode
        return at

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

    def __build_local_nodes_2D(self):
        """Builds local element nodes, locations and I,J,K numbering"""
        degree = self.degree
        if degree == 1:
            matrix_IJK = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        elif degree == 2:
            matrix_IJK = np.array(
                [[0, 0, 2], [0, 2, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
            )
        elif degree == 3:
            matrix_IJK = np.array(
                [
                    [0, 0, 3],
                    [0, 3, 0],
                    [3, 0, 0],
                    [1, 2, 0],
                    [2, 1, 0],
                    [1, 0, 2],
                    [2, 0, 1],
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 1, 1],
                ]
            )

        elif degree == 4:
            matrix_IJK = np.array(
                [
                    [0, 0, 4],
                    [0, 4, 0],
                    [4, 0, 0],
                    [1, 3, 0],
                    [2, 2, 0],
                    [3, 1, 0],
                    [1, 0, 3],
                    [2, 0, 2],
                    [3, 0, 1],
                    [0, 1, 3],
                    [0, 2, 2],
                    [0, 3, 1],
                    [1, 1, 2],
                    [1, 2, 1],
                    [2, 1, 1],
                ]
            )

        elif degree == 5:
            matrix_IJK = np.array(
                [
                    [0, 0, 5],
                    [0, 5, 0],
                    [5, 0, 0],
                    [1, 4, 0],
                    [2, 3, 0],
                    [3, 2, 0],
                    [4, 1, 0],
                    [1, 0, 4],
                    [2, 0, 3],
                    [3, 0, 2],
                    [4, 0, 1],
                    [0, 1, 4],
                    [0, 2, 3],
                    [0, 3, 2],
                    [0, 4, 1],
                    [1, 1, 3],
                    [1, 2, 2],
                    [1, 3, 1],
                    [2, 1, 2],
                    [2, 2, 1],
                    [3, 1, 1],
                ]
            )

        elif degree == 6:
            matrix_IJK = np.array(
                [
                    [0, 0, 6],
                    [0, 6, 0],
                    [6, 0, 0],
                    [1, 5, 0],
                    [2, 4, 0],
                    [3, 3, 0],
                    [4, 2, 0],
                    [5, 1, 0],
                    [1, 0, 5],
                    [2, 0, 4],
                    [3, 0, 3],
                    [4, 0, 2],
                    [5, 0, 1],
                    [0, 1, 5],
                    [0, 2, 4],
                    [0, 3, 3],
                    [0, 4, 2],
                    [0, 5, 1],
                    [1, 1, 4],
                    [1, 2, 3],
                    [1, 3, 2],
                    [1, 4, 1],
                    [2, 1, 3],
                    [2, 2, 2],
                    [2, 3, 1],
                    [3, 1, 2],
                    [3, 2, 1],
                    [4, 1, 1],
                ]
            )

        elif degree > 6:
            mesh = UnitSquareMesh(1, 1)
            xmesh, ymesh = SpatialCoordinate(mesh)
            V = FunctionSpace(mesh, "CG", degree)
            u = Function(V).interpolate(xmesh)
            x = u.dat.data[:]

            u = Function(V).interpolate(ymesh)
            y = u.dat.data[:]

            # Getting vetor that shows dof of each node
            fdrake_cell_node_map = V.cell_node_map()
            cell_node_map = fdrake_cell_node_map.values
            local_nodes = cell_node_map[
                0:2
            ]  # first 3 are vertices, then sides, then interior following a diagonal

            matrix_IJK = np.zeros((len(local_nodes), 3))
            TOL = 1e-6

            cont_aux = 0
            for node in local_nodes:
                # Finding I
                I = degree - (x[node] + y[node] + TOL) // (1 / degree)
                # Finding J
                J = (x[node] + TOL) // (1 / degree)
                # Fingind K
                K = (y[node] + TOL) // (1 / degree)

                matrix_IJK[cont_aux, :] = [I, J, K]
                cont_aux += 1
        else:
            raise ValueError("Degree is not supported by the interpolator")

        return matrix_IJK

    def __build_local_nodes_3D(self):
        """Builds local element nodes, locations and I,J,K numbering"""
        degree = self.degree
        if degree == 1:
            matrix_IJK = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        elif degree == 2:
            matrix_IJK = np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            )

        elif degree == 3:
            matrix_IJK = np.array(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, 2.0, 1.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 2.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [2.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0],
                    [2.0, 1.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                ]
            )

        elif degree == 4:
            matrix_IJK = np.array(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                    [0.0, 0.0, 3.0, 1.0],
                    [0.0, 0.0, 2.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 3.0, 0.0, 1.0],
                    [0.0, 2.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 3.0],
                    [0.0, 3.0, 1.0, 0.0],
                    [0.0, 2.0, 2.0, 0.0],
                    [0.0, 1.0, 3.0, 0.0],
                    [3.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 2.0],
                    [1.0, 0.0, 0.0, 3.0],
                    [3.0, 0.0, 1.0, 0.0],
                    [2.0, 0.0, 2.0, 0.0],
                    [1.0, 0.0, 3.0, 0.0],
                    [3.0, 1.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0, 0.0],
                    [1.0, 3.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0, 1.0],
                    [0.0, 1.0, 2.0, 1.0],
                    [0.0, 1.0, 1.0, 2.0],
                    [2.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0, 2.0],
                    [2.0, 1.0, 0.0, 1.0],
                    [1.0, 2.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 2.0],
                    [2.0, 1.0, 1.0, 0.0],
                    [1.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 2.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )

        elif degree == 5:
            matrix_IJK = np.array(
                [
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 4.0, 1.0],
                    [0.0, 0.0, 3.0, 2.0],
                    [0.0, 0.0, 2.0, 3.0],
                    [0.0, 0.0, 1.0, 4.0],
                    [0.0, 4.0, 0.0, 1.0],
                    [0.0, 3.0, 0.0, 2.0],
                    [0.0, 2.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 4.0],
                    [0.0, 4.0, 1.0, 0.0],
                    [0.0, 3.0, 2.0, 0.0],
                    [0.0, 2.0, 3.0, 0.0],
                    [0.0, 1.0, 4.0, 0.0],
                    [4.0, 0.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 2.0],
                    [2.0, 0.0, 0.0, 3.0],
                    [1.0, 0.0, 0.0, 4.0],
                    [4.0, 0.0, 1.0, 0.0],
                    [3.0, 0.0, 2.0, 0.0],
                    [2.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 4.0, 0.0],
                    [4.0, 1.0, 0.0, 0.0],
                    [3.0, 2.0, 0.0, 0.0],
                    [2.0, 3.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 0.0],
                    [0.0, 3.0, 1.0, 1.0],
                    [0.0, 2.0, 2.0, 1.0],
                    [0.0, 1.0, 3.0, 1.0],
                    [0.0, 2.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0, 2.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [3.0, 0.0, 1.0, 1.0],
                    [2.0, 0.0, 2.0, 1.0],
                    [1.0, 0.0, 3.0, 1.0],
                    [2.0, 0.0, 1.0, 2.0],
                    [1.0, 0.0, 2.0, 2.0],
                    [1.0, 0.0, 1.0, 3.0],
                    [3.0, 1.0, 0.0, 1.0],
                    [2.0, 2.0, 0.0, 1.0],
                    [1.0, 3.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0, 2.0],
                    [1.0, 2.0, 0.0, 2.0],
                    [1.0, 1.0, 0.0, 3.0],
                    [3.0, 1.0, 1.0, 0.0],
                    [2.0, 2.0, 1.0, 0.0],
                    [1.0, 3.0, 1.0, 0.0],
                    [2.0, 1.0, 2.0, 0.0],
                    [1.0, 2.0, 2.0, 0.0],
                    [1.0, 1.0, 3.0, 0.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                ]
            )

        elif degree == 6:
            matrix_IJK = np.array(
                [
                    [6.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 0.0],
                    [0.0, 0.0, 0.0, 6.0],
                    [0.0, 0.0, 5.0, 1.0],
                    [0.0, 0.0, 4.0, 2.0],
                    [0.0, 0.0, 3.0, 3.0],
                    [0.0, 0.0, 2.0, 4.0],
                    [0.0, 0.0, 1.0, 5.0],
                    [0.0, 5.0, 0.0, 1.0],
                    [0.0, 4.0, 0.0, 2.0],
                    [0.0, 3.0, 0.0, 3.0],
                    [0.0, 2.0, 0.0, 4.0],
                    [0.0, 1.0, 0.0, 5.0],
                    [0.0, 5.0, 1.0, 0.0],
                    [0.0, 4.0, 2.0, 0.0],
                    [0.0, 3.0, 3.0, 0.0],
                    [0.0, 2.0, 4.0, 0.0],
                    [0.0, 1.0, 5.0, 0.0],
                    [5.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 2.0],
                    [3.0, 0.0, 0.0, 3.0],
                    [2.0, 0.0, 0.0, 4.0],
                    [1.0, 0.0, 0.0, 5.0],
                    [5.0, 0.0, 1.0, 0.0],
                    [4.0, 0.0, 2.0, 0.0],
                    [3.0, 0.0, 3.0, 0.0],
                    [2.0, 0.0, 4.0, 0.0],
                    [1.0, 0.0, 5.0, 0.0],
                    [5.0, 1.0, 0.0, 0.0],
                    [4.0, 2.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0, 0.0],
                    [2.0, 4.0, 0.0, 0.0],
                    [1.0, 5.0, 0.0, 0.0],
                    [0.0, 4.0, 1.0, 1.0],
                    [0.0, 3.0, 2.0, 1.0],
                    [0.0, 2.0, 3.0, 1.0],
                    [0.0, 1.0, 4.0, 1.0],
                    [0.0, 3.0, 1.0, 2.0],
                    [0.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 2.0],
                    [0.0, 2.0, 1.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 1.0, 4.0],
                    [4.0, 0.0, 1.0, 1.0],
                    [3.0, 0.0, 2.0, 1.0],
                    [2.0, 0.0, 3.0, 1.0],
                    [1.0, 0.0, 4.0, 1.0],
                    [3.0, 0.0, 1.0, 2.0],
                    [2.0, 0.0, 2.0, 2.0],
                    [1.0, 0.0, 3.0, 2.0],
                    [2.0, 0.0, 1.0, 3.0],
                    [1.0, 0.0, 2.0, 3.0],
                    [1.0, 0.0, 1.0, 4.0],
                    [4.0, 1.0, 0.0, 1.0],
                    [3.0, 2.0, 0.0, 1.0],
                    [2.0, 3.0, 0.0, 1.0],
                    [1.0, 4.0, 0.0, 1.0],
                    [3.0, 1.0, 0.0, 2.0],
                    [2.0, 2.0, 0.0, 2.0],
                    [1.0, 3.0, 0.0, 2.0],
                    [2.0, 1.0, 0.0, 3.0],
                    [1.0, 2.0, 0.0, 3.0],
                    [1.0, 1.0, 0.0, 4.0],
                    [4.0, 1.0, 1.0, 0.0],
                    [3.0, 2.0, 1.0, 0.0],
                    [2.0, 3.0, 1.0, 0.0],
                    [1.0, 4.0, 1.0, 0.0],
                    [3.0, 1.0, 2.0, 0.0],
                    [2.0, 2.0, 2.0, 0.0],
                    [1.0, 3.0, 2.0, 0.0],
                    [2.0, 1.0, 3.0, 0.0],
                    [1.0, 2.0, 3.0, 0.0],
                    [1.0, 1.0, 4.0, 0.0],
                    [3.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, 1.0],
                    [1.0, 3.0, 1.0, 1.0],
                    [2.0, 1.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 1.0, 3.0, 1.0],
                    [2.0, 1.0, 1.0, 2.0],
                    [1.0, 2.0, 1.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 3.0],
                ]
            )

        elif degree > 6:
            mesh = UnitTetrahedronMesh()
            xmesh, ymesh, zmesh = SpatialCoordinate(mesh)
            V = FunctionSpace(mesh, "CG", degree)
            u = Function(V).interpolate(xmesh)
            x = u.dat.data[:]

            u = Function(V).interpolate(ymesh)
            y = u.dat.data[:]

            u = Function(V).interpolate(zmesh)
            z = u.dat.data[:]

            # Getting vetor that shows dof of each node
            fdrake_cell_node_map = V.cell_node_map()
            cell_node_map = fdrake_cell_node_map.values
            local_nodes = cell_node_map[
                0, :
            ]  # first 3 are vertices, then sides, then interior following a diagonal

            matrix_IJK = np.zeros((len(local_nodes), 4))
            TOL = 1e-6

            cont_aux = 0
            for node in local_nodes:
                # Finding I
                I = degree - (x[node] + y[node] + z[node] + TOL) // (1 / degree)
                # Finding J
                J = (x[node] + TOL) // (1 / degree)
                # Fingind K
                K = (y[node] + TOL) // (1 / degree)
                # Finding Q
                Q = (z[node] + TOL) // (1 / degree)

                matrix_IJK[cont_aux, :] = [I, J, K, Q]
                cont_aux += 1
        else:
            raise ValueError("degree is not supported by the interpolator")

        return matrix_IJK


# End of class, some helper functions


def _lagrange_basis_1d(x, p, P, zeros):
    """Builds a simple Lagrange basis function

    Parameters
    x: location to evaluate basis function
    p: current degree
    P: Overall degree of function space
    zeros: zeros to be used in this basis
        function (equispaced, GLL, or etc.)
    """
    p = int(p)
    P = int(P)
    value = 1
    for q in range(P + 1):
        if q != p:
            value *= (x - zeros[q]) / (zeros[p] - zeros[q])
    return value


def triangle_area(p1, p2, p3):
    """Calculate the area of a triangle.

    Parameters
    ----------

    p1: 1st vertex of the triangle
    p2: 2nd vertex of the triangle
    p3: 3rd vertex of the triangle

    Returns
    -------
    Triangle area
    """
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def tetrahedral_volume(p1, p2, p3, p4):
    """Calculate the volume of a tetrahedral."""
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3
    (x4, y4, z4) = p4

    A = np.array([x1, y1, z1])
    B = np.array([x2, y2, z2])
    C = np.array([x3, y3, z3])
    D = np.array([x4, y4, z4])

    return abs(1.0 / 6.0 * (np.dot(B - A, np.cross(C - A, D - A))))
