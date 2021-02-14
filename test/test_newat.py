import math
import numpy as np
from copy import deepcopy

from firedrake import *

import spyro

from .inputfiles.Model1_2d_CG import model
from .inputfiles.Model1_3d_CG import model as model3D


def triangle_area(p1, p2, p3):
    """Simple function to calculate triangle area based on its 3 vertices."""
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3

    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def test_correct_receiver_location_generation2D():
    """Tests if receiver locations where generated correctly"""
    comm = spyro.utils.mpi_init(model)
    mesh, V = spyro.io.read_mesh(model, comm)

    receivers = spyro.create_transect((-0.1, 0.3), (-0.1, 0.9), 3)
    answer = np.array([[-0.1, 0.3], [-0.1, 0.6], [-0.1, 0.9]])

    assert np.allclose(receivers, answer)


def test_correct_receiver_to_cell_location2D():
    """Tests if the receivers where located in the correct cell"""
    comm = spyro.utils.mpi_init(model)
    model["opts"]["degree"] = 3
    mesh, V = spyro.io.read_mesh(model, comm)

    model["acquisition"]["num_receivers"] = 3
    recvs = spyro.create_transect((-0.1, 0.3), (-0.1, 0.9), 3)
    recvs = model["acquisition"]["receiver_locations"] = recvs

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    # test 1
    cell_vertex1 = receivers.cellVertices[0][0]
    cell_vertex2 = receivers.cellVertices[0][1]
    cell_vertex3 = receivers.cellVertices[0][2]
    x = receivers.receiver_locations[0, 0]
    y = receivers.receiver_locations[0, 1]
    p = (x, y)

    areaT = triangle_area(cell_vertex1, cell_vertex2, cell_vertex3)
    area1 = triangle_area(p, cell_vertex2, cell_vertex3)
    area2 = triangle_area(cell_vertex1, p, cell_vertex3)
    area3 = triangle_area(cell_vertex1, cell_vertex2, p)

    test1 = math.isclose((area1 + area2 + area3), areaT, rel_tol=1e-09)

    # test 2
    cell_vertex1 = receivers.cellVertices[1][0]
    cell_vertex2 = receivers.cellVertices[1][1]
    cell_vertex3 = receivers.cellVertices[1][2]
    x = receivers.receiver_locations[1, 0]
    y = receivers.receiver_locations[1, 1]
    p = (x, y)

    areaT = triangle_area(cell_vertex1, cell_vertex2, cell_vertex3)
    area1 = triangle_area(p, cell_vertex2, cell_vertex3)
    area2 = triangle_area(cell_vertex1, p, cell_vertex3)
    area3 = triangle_area(cell_vertex1, cell_vertex2, p)

    test2 = math.isclose((area1 + area2 + area3), areaT, rel_tol=1e-09)

    # test 3
    cell_vertex1 = receivers.cellVertices[2][0]
    cell_vertex2 = receivers.cellVertices[2][1]
    cell_vertex3 = receivers.cellVertices[2][2]
    x = receivers.receiver_locations[2, 0]
    y = receivers.receiver_locations[2, 1]
    p = (x, y)

    areaT = triangle_area(cell_vertex1, cell_vertex2, cell_vertex3)
    area1 = triangle_area(p, cell_vertex2, cell_vertex3)
    area2 = triangle_area(cell_vertex1, p, cell_vertex3)
    area3 = triangle_area(cell_vertex1, cell_vertex2, p)

    test3 = math.isclose((area1 + area2 + area3), areaT, rel_tol=1e-09)

    assert all([test1, test2, test3])


def test_correct_at_value2D():
    comm = spyro.utils.mpi_init(model)
    model["opts"]["degree"] = 3
    mesh, V = spyro.io.read_mesh(model, comm)
    pz = -0.1
    px = 0.3
    recvs = spyro.create_transect(
        (pz, px), (pz, px), 3
    )
    #recvs = spyro.create_transect(
    #    (-0.00935421,  3.25160664), (-0.00935421,  3.25160664), 3
    #)
    model["acquisition"]["receiver_locations"] = recvs
    model["acquisition"]["num_receivers"] = 3

    receivers = spyro.Receivers(model, mesh, V, comm).create()
    V = receivers.space
    z, x = SpatialCoordinate(mesh)

    u1 = Function(V).interpolate(x + z)
    test1 = math.isclose(
        (pz+px), receivers._Receivers__new_at(u1.dat.data[:], 0, True), rel_tol=1e-09
    )

    u1 = Function(V).interpolate(sin(x) * z * 2)
    test2 = math.isclose(
        sin(px) * pz * 2,
        receivers._Receivers__new_at(u1.dat.data[:], 0, True),
        rel_tol=1e-05,
    )

    assert all([test1, test2])


def tetrahedral_volume(p1, p2, p3, p4):
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3
    (x4, y4, z4) = p4

    A = np.array([x1, y1, z1])
    B = np.array([x2, y2, z2])
    C = np.array([x3, y3, z3])
    D = np.array([x4, y4, z4])

    volume = abs(1.0 / 6.0 * (np.dot(B - A, np.cross(C - A, D - A))))

    return volume


def test_correct_receiver_location_generation3D():
    """Tests if receiver locations where generated correctly"""

    test_model = deepcopy(model3D)
    comm = spyro.utils.mpi_init(test_model)
    mesh, V = spyro.io.read_mesh(test_model, comm)
    test_model["acquisition"]["num_receivers"] = 3
    receivers = spyro.create_transect((-0.05, 0.3, 0.5), (-0.05, 0.9, 0.5), 3)
    test_model["acquisition"]["receiver_locations"] = receivers
    receivers = spyro.Receivers(test_model, mesh, V, comm).create()
    answer = np.array([[-0.05, 0.3, 0.5], [-0.05, 0.6, 0.5], [-0.05, 0.9, 0.5]])

    assert np.allclose(receivers.receiver_locations, answer)


def test_correct_receiver_to_cell_location3D():
    """Tests if the receivers where located in the correct cell"""

    test_model1 = deepcopy(model3D)
    comm = spyro.utils.mpi_init(test_model1)
    mesh, V = spyro.io.read_mesh(test_model1, comm)
    rec = spyro.create_transect((-0.05, 0.1, 0.5), (-0.05, 0.9, 0.5), 3)
    test_model1["acquisition"]["receiver_locations"] = rec
    test_model1["acquisition"]["num_receivers"] = 3
    receivers = spyro.Receivers(test_model1, mesh, V, comm).create()

    # test 1
    cell_vertex1 = receivers.cellVertices[0][0]
    cell_vertex2 = receivers.cellVertices[0][1]
    cell_vertex3 = receivers.cellVertices[0][2]
    cell_vertex4 = receivers.cellVertices[0][3]
    x = receivers.receiver_locations[0, 0]
    y = receivers.receiver_locations[0, 1]
    z = receivers.receiver_locations[0, 2]
    p = (x, y, z)

    volumeT = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, cell_vertex4)
    volume1 = tetrahedral_volume(p, cell_vertex2, cell_vertex3, cell_vertex4)
    volume2 = tetrahedral_volume(cell_vertex1, p, cell_vertex3, cell_vertex4)
    volume3 = tetrahedral_volume(cell_vertex1, cell_vertex2, p, cell_vertex4)
    volume4 = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, p)

    test1 = math.isclose(
        (volume1 + volume2 + volume3 + volume4), volumeT, rel_tol=1e-09
    )

    # test 2
    cell_vertex1 = receivers.cellVertices[1][0]
    cell_vertex2 = receivers.cellVertices[1][1]
    cell_vertex3 = receivers.cellVertices[1][2]
    cell_vertex4 = receivers.cellVertices[1][3]
    x = receivers.receiver_locations[1, 0]
    y = receivers.receiver_locations[1, 1]
    z = receivers.receiver_locations[1, 2]
    p = (x, y, z)

    volumeT = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, cell_vertex4)
    volume1 = tetrahedral_volume(p, cell_vertex2, cell_vertex3, cell_vertex4)
    volume2 = tetrahedral_volume(cell_vertex1, p, cell_vertex3, cell_vertex4)
    volume3 = tetrahedral_volume(cell_vertex1, cell_vertex2, p, cell_vertex4)
    volume4 = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, p)

    test2 = math.isclose(
        (volume1 + volume2 + volume3 + volume4), volumeT, rel_tol=1e-09
    )

    # test 3
    cell_vertex1 = receivers.cellVertices[2][0]
    cell_vertex2 = receivers.cellVertices[2][1]
    cell_vertex3 = receivers.cellVertices[2][2]
    cell_vertex4 = receivers.cellVertices[2][3]
    x = receivers.receiver_locations[2, 0]
    y = receivers.receiver_locations[2, 1]
    z = receivers.receiver_locations[2, 2]
    p = (x, y, z)

    volumeT = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, cell_vertex4)
    volume1 = tetrahedral_volume(p, cell_vertex2, cell_vertex3, cell_vertex4)
    volume2 = tetrahedral_volume(cell_vertex1, p, cell_vertex3, cell_vertex4)
    volume3 = tetrahedral_volume(cell_vertex1, cell_vertex2, p, cell_vertex4)
    volume4 = tetrahedral_volume(cell_vertex1, cell_vertex2, cell_vertex3, p)

    test3 = math.isclose(
        (volume1 + volume2 + volume3 + volume4), volumeT, rel_tol=1e-09
    )

    assert all([test1, test2, test3])


def test_correct_at_value3D():
    test_model2 = deepcopy(model3D)
    test_model2["acquisition"]["num_receivers"] = 3
    test_model2["opts"]["degree"] = 3
    comm = spyro.utils.mpi_init(test_model2)
    mesh, V = spyro.io.read_mesh(test_model2, comm)
    x_start = 0.09153949331982138
    x_end = 0.09153949331982138
    z_start = 0.0
    z_end = 0.0
    y_start = 0.47342699605572036
    y_end = 0.47342699605572036

    x_real, y_real, z_real = x_start, y_start, z_start

    recvs = spyro.create_transect(
        (z_start, x_start, y_start), (z_end, x_end, y_end), 3
    )
    test_model2["acquisition"]["receiver_locations"] = recvs
    receivers = spyro.Receivers(test_model2, mesh, V, comm).create()
    V = receivers.space
    z, x, y = SpatialCoordinate(mesh)

    u1 = Function(V).interpolate(x + z + y)
    realvalue = x_real + y_real + z_real
    test1 = math.isclose(
        realvalue, receivers._Receivers__new_at(u1.dat.data[:], 0, True), rel_tol=1e-09
    )

    u1 = Function(V).interpolate(sin(x) * (z + 1) ** 2 * cos(y))
    realvalue = sin(x_real) * (z_real + 1) ** 2 * cos(y_real)
    test2 = math.isclose(
        realvalue, receivers._Receivers__new_at(u1.dat.data[:], 0, True), rel_tol=1e-05
    )

    assert all([test1, test2])


if __name__ == "__main__":
    test_correct_receiver_location_generation2D()
    test_correct_receiver_to_cell_location2D()
    test_correct_at_value2D()
    test_correct_receiver_location_generation3D()
    test_correct_receiver_to_cell_location3D()
    test_correct_at_value3D()
