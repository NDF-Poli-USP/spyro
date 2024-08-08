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
from .changing_coordinates import (
    change_to_reference_triangle,
    change_to_reference_tetrahedron,
    change_to_reference_quad,
    change_to_reference_hexa,
)

import numpy as np


class Delta_projector:
    """
    Class that interpolates the solution to the receiver coordinates

    Attributes
    ----------
    mesh: firedrake.Mesh
        Mesh object
    space: firedrake.FunctionSpace
        Function space to be used
    my_ensemble: mpi4py.MPI.Intracomm
        MPI communicator
    dimension: int
        Dimension of the mesh
    degree: int
        Degree of FEM space
    point_locations: list
        List of tuples of point locations
    number_of_points: int
        Number of points
    cellIDs: list
        List of cell IDs for each point
    cellVertices: list
        List of vertices for each cell containing a point
    cell_tabulations: list
        List of tabulations for each point in a cell
    cellNodeMaps: list
        List of node maps for each cell
    nodes_per_cell: int
        Number of nodes per cell
    quadrilateral: bool
        True if mesh is quadrilateral
    is_local: list
        List of cell IDs local to the processor
    """
    def __init__(self, wave_object):
        """
        Initializes the class

        Parameters
        ----------
        wave_object: spyro.wave.Wave
            Wave object
        """
        self.automatic_adjoint = wave_object.automatic_adjoint
        self.mesh = wave_object.mesh
        self.space = wave_object.function_space.sub(0)
        self.my_ensemble = wave_object.comm
        self.dimension = wave_object.dimension
        self.degree = wave_object.degree

        self.point_locations = None
        self.number_of_points = None

        self.cellIDs = None
        self.cellVertices = None
        self.cell_tabulations = None
        self.cellNodeMaps = None
        self.nodes_per_cell = None
        if wave_object.cell_type == "quadrilateral":
            self.quadrilateral = True
        else:
            self.quadrilateral = False
        self.is_local = None

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
            cell_id = self.mesh.locate_cell(self.point_locations[rid],
                                            tolerance=1e-6)
            self.is_local[rid] = cell_id

        (
            self.cellIDs,
            self.cellVertices,
            self.cellNodeMaps,
        ) = self.__point_locator()
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
        return [self.new_at(field, rn) for rn in range(self.number_of_points)]

    def new_at(self, udat, receiver_id):
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

        cell_tabulations = np.zeros(
            (self.number_of_points, self.nodes_per_cell)
        )

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                cell_vertices = [v0, v1, v2]

                p_reference = change_to_reference_triangle(p, cell_vertices)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def __func_build_cell_tabulations_3D(self):
        element = choosing_element(self.space, self.degree)

        cell_tabulations = np.zeros(
            (self.number_of_points, self.nodes_per_cell)
        )

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]
                cell_vertices = [v0, v1, v2, v3]

                p_reference = change_to_reference_tetrahedron(p, cell_vertices)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

    def __func_build_cell_tabulations_2D_quad(self):
        # finatelement = FiniteElement('CG', self.mesh.ufl_cell(),
        # degree=self.degree, variant='spectral')
        V = self.space

        element = V.finat_element.fiat_equivalent

        cell_tabulations = np.zeros(
            (self.number_of_points, self.nodes_per_cell)
        )

        for receiver_id in range(self.number_of_points):
            cell_id = self.is_local[receiver_id]
            if cell_id is not None:
                # getting coordinates to change to reference element
                p = self.point_locations[receiver_id]
                v0 = self.cellVertices[receiver_id][0]
                v1 = self.cellVertices[receiver_id][1]
                v2 = self.cellVertices[receiver_id][2]
                v3 = self.cellVertices[receiver_id][3]
                cell_vertices = [v0, v1, v2, v3]

                p_reference = change_to_reference_quad(p, cell_vertices)
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

        cell_tabulations = np.zeros(
            (self.number_of_points, self.nodes_per_cell)
        )

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
                cell_vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

                p_reference = change_to_reference_hexa(p, cell_vertices)
                initial_tab = element.tabulate(0, [p_reference])
                phi_tab = initial_tab[(0, 0, 0)]

                cell_tabulations[receiver_id, :] = phi_tab.transpose()

        return cell_tabulations

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

    def __point_locator(self):
        """Function that returns a list of tuples and a matrix
        the list of tuples has in line n the receiver position
        and the position of the nodes in the element that contains
        the receiver.
        The matrix has the deegres of freedom of the nodes inside
        same element as the receiver.
        """
        if self.dimension == 2:
            return self.__point_locator_2D()
        elif self.dimension == 3:
            return self.__point_locator_3D()
        else:
            raise ValueError

    def __point_locator_2D(self):
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

    def __point_locator_3D(self):
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

        return cellId_maps, cellVertices, cellNodeMaps


def choosing_geometry(cell_geometry):
    """
    Chooses UFC reference element geometry based on desired function space

    Parameters
    ----------
    cell_geometry : firedrake.Cell
        Cell geometry of the mesh.

    Returns
    -------
    T : FIAT reference element
        FIAT reference element to be used in the interpolation.
    """
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

    return T


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
    T = choosing_geometry(V.mesh().ufl_cell())

    if V.ufl_element().family() == "Kong-Mulder-Veldhuizen":
        element = KMV(T, degree)
    elif V.ufl_element().family() == "Lagrange":
        element = CG(T, degree)
    elif V.ufl_element().family() == "Discontinuous Lagrange":
        element = DG(T, degree)
    else:
        raise ValueError("Function space not yet supported.")

    return element


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
