import firedrake as fire
import math
import pytest
import spyro
try:
    from SeismicMesh import write_velocity_model
    seismic_mesh = True
except ImportError:
    seismic_mesh = False


@pytest.mark.skipif(not seismic_mesh, reason="No SeismicMesh")
def test_read_and_write_segy():
    vp_name = "velocity_models/test"
    segy_file = vp_name+".segy"
    hdf5_file = vp_name+".hdf5"
    mesh = fire.UnitSquareMesh(10, 10)
    mesh.coordinates.dat.data[:, 0] *= -1

    V = fire.FunctionSpace(mesh, 'CG', 3)
    x, y = fire.SpatialCoordinate(mesh)
    r = 0.2
    xc = -0.5
    yc = 0.5

    vp = fire.Function(V)

    c = fire.conditional((x-xc)**2+(y-yc)**2 < r**2, 3.0, 1.5)

    vp.interpolate(c)

    xi, yi, zi = spyro.io.write_function_to_grid(vp, V, 10.0/1000.0)
    spyro.io.create_segy(zi, segy_file)
    write_velocity_model(segy_file, vp_name)

    model = {}

    model["opts"] = {
        "method": "CG",  # either CG or KMV
        "quadrature": "CG",  # Equi or KMV
        "degree": 3,  # p order
        "dimension": 2,  # dimension
    }
    model["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": None,
        "initmodel": None,
        "truemodel": hdf5_file,
    }
    model["BCs"] = {
        "status": False,
    }

    vp_read = spyro.io.interpolate(model, mesh, V, guess=False)

    fire.File("velocity_models/test.pvd").write(vp_read)

    value_at_center = vp_read.at(xc, yc)
    test1 = math.isclose(value_at_center, 3.0)
    value_outside_circle = vp_read.at(xc+r+0.1, yc)
    test2 = math.isclose(value_outside_circle, 1.5)
    assert all([test1, test2])


if __name__ == "__main__":
    test_read_and_write_segy()
