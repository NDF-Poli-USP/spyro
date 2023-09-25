import firedrake as fire
import numpy as np
from scipy.interpolate import griddata
import warnings


def loading_csv_into_function(V, file_name, crossed=False):
    print("a")
    dimension = V.ufl_domain()._geometric_dimension
    mesh = V.ufl_domain()

    # TODO change columns for 3D
    field = np.loadtxt(file_name, delimiter=",", skiprows=1, usecols=(0, 1, 2))
    Lx = np.amax(field[:, 1])
    Lz = np.amax(field[:, 2])

    # GEtting correct Lz if negative z values
    Lzmin = np.amin(field[:, 1])
    if np.abs(Lzmin) > np.abs(Lz):
        Lz = Lzmin
    if np.abs(Lz) > 500 or np.abs(Lx) > 500:
        warnings.warn("Assuming m/s changing to km/s")
        field[:, 1] = field[:, 1] / 1000
        field[:, 2] = field[:, 2] / 1000
    if Lz > 0:
        field[:, 2] = field[:, 2]*(-1)
    
    # if dimension == 2:

    # print("END")
    # points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
    #     Data point coordinates.
    # values : ndarray of float or complex, shape (n,)
    #     Data values.
    W = fire.VectorFunctionSpace(mesh, V.ufl_element())
    coords = fire.interpolate(mesh.coordinates, W)
    z_mesh, x_mesh = coords.dat.data[:, 0], coords.dat.data[:, 1]
    points = np.zeros(np.shape(field[:, 1:]))
    points[:, 0] = field[:, 2]
    points[:, 1] = field[:, 1]
    values = field[:, 0]

    new_field = griddata(points, values, (z_mesh, x_mesh))
    u = fire.Function(V)
    u.dat.data[:] = new_field
    out = fire.File("testingHLIN.pvd")
    out.write(u)

    # Revisar valores menores que raiz(2)/2 para o CosHig

    # # Domain shape depending on format
    # shape = (nx+1, ny+1)
    # prop = prop.reshape(np.flipud(shape))
    # # Get field with indexes in the same order as vertices from function
    # prop = np.flipud(prop).flat


if __name__ == "__main__":
    print("a")
    Lz = 3.0188
    Lx = 6.0375
    mesh = fire.RectangleMesh(100, 100, Lz, Lx, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V)
    # loading_csv_into_function(V, "cosHig.csv")
    loading_csv_into_function(V, "InfHLIN.csv")
