import firedrake as fire
import SeismicMesh as sm
import numpy as np

def RectangleMesh(Lx, Ly, nx, ny, pad = None, comm=None, quadrilateral=False):
    """Create a rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both axis are dislocated by the pad.
    
    Parameters
    ----------
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
        MPI communicator. The default is None.
    quadrilateral : bool, optional
        If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
        Mesh
    """
    if pad != None:
        Lx += pad
        Ly += 2*pad
    else:
        pad = 0
    mesh = fire.RectangleMesh(Lx, Ly, nx, ny, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh

def PeriodicRectangleMesh(Lx, Ly, nx, ny, pad = None, comm=None, quadrilateral=False):
    """Create a periodic rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both axis are dislocated by the pad.

    Parameters
    ----------
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
        MPI communicator. The default is None.
    quadrilateral : bool, optional
        If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
        Mesh
    
    """
    if pad != None:
        Lx += pad
        Ly += 2*pad
    else:
        pad = 0
    mesh = fire.PeriodicRectangleMesh(Lx, Ly, nx, ny, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh

def UnitSquareMesh(nx, ny, quadrilateral=False, comm=None):
    mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=quadrilateral, reorder=reorder, distribution_parameters=distribution_parameters, comm=comm, **kwargs)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    return mesh





