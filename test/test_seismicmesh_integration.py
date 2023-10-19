import spyro
import firedrake as fire
import numpy as np


def mean_edge_length(triangle):
    """
    Compute the mean edge length of a triangle
    """
    (x0, y0), (x1, y1), (x2, y2) = triangle
    l0 = np.sqrt((x1-x0)**2+(y1-y0)**2)
    l1 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    l2 = np.sqrt((x0-x2)**2+(y0-y2)**2)
    return (l0+l1+l2)/3.0


def test_spyro_seimicmesh_2d_homogeneous_generation():
    Lz = 1.0
    Lx = 2.0
    c = 1.5
    freq = 5.0
    lbda = c/freq
    pad = 0.3
    cpw = 3

    mesh_parameters = {
        "length_z": Lz,
        "length_x": Lx,
        "length_y": 0.0,
        "cell_type": "triangle",
        "mesh_type": "SeismicMesh",
        "dx": None,
        "periodic": False,
        "velocity_model_file": None,
        "cells_per_wavelength": cpw,
        "source_frequency": freq,
        "minimum_velocity": c,
        "abc_pad_length": pad,
        "lbda": lbda,
        "dimension": 2,
        "edge_length": lbda/cpw,
    }

    Mesh_obj = spyro.meshing.AutomaticMesh(
        mesh_parameters=mesh_parameters,
    )
    Mesh_obj.set_seismicmesh_parameters(output_file_name="test.msh")

    mesh = Mesh_obj.create_mesh()

    V = fire.FunctionSpace(mesh, "CG", 1)
    z_mesh, x_mesh = fire.SpatialCoordinate(mesh)
    uz = fire.Function(V).interpolate(z_mesh)
    ux = fire.Function(V).interpolate(x_mesh)

    z = uz.dat.data[:]
    x = ux.dat.data[:]

    # Testing if boundaries are correct
    test1 = (np.isclose(np.amin(z), -Lz-pad))
    test1 = test1 and (np.isclose(np.amax(x), Lx+pad))
    test1 = test1 and (np.isclose(np.amax(z), 0.0))
    test1 = test1 and (np.isclose(np.amin(x), -pad))
    print(f"Boundary values are correct: {test1}")

    # Checking edge length of an interior cell
    node_ids = V.cell_node_list[300]
    p0 = (z[node_ids[0]], x[node_ids[0]])
    p1 = (z[node_ids[1]], x[node_ids[1]])
    p2 = (z[node_ids[2]], x[node_ids[2]])

    le = mean_edge_length((p0, p1, p2))
    le_expected = lbda/cpw
    test2 = np.isclose(le, le_expected, rtol=1e-1)
    print(f"Edge length is correct: {test2}")

    assert all([test1, test2])


if __name__ == "__main__":
    test_spyro_seimicmesh_2d_homogeneous_generation()
