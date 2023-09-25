import firedrake as fire
import numpy as np
import warnings


def loading_csv_into_function(firedrake_function, file_name):
    print("a")
    V = firedrake_function.function_space()
    # change columns for 3D
    field = np.loadtxt(file_name, delimiter=",", skiprows=1, usecols=(0, 1, 2))
    Lz = np.amax(field[:, 1])
    Lx = np.amax(field[:, 2])
    if Lz > 500 or Lx > 500:
        warnings.warn("Assuming m/s changing to km/s")
        field[:, 1] = field[:, 1] / 1000
        field[:, 2] = field[:, 2] / 1000
    
    print("END")


if __name__ == "__main__":
    print("a")
    mesh = fire.UnitSquareMesh(10, 10, quadrilateral=True)
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V)
    loading_csv_into_function(u, "cosHig.csv")
