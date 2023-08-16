import firedrake as fire


class AutomaticMesh:
    def __init__(self, dimension=2, comm=None, abc_pad=None):
        """
        Parameters
        ----------
        dimension : int, optional
            Dimension of the mesh. The default is 2.
        comm : MPI communicator, optional
            MPI communicator. The default is None.
        """
        self.dimension = dimension
        self.length_z = None
        self.length_x = None
        self.length_y = None
        self.dx = None
        self.quadrilateral = False
        self.periodic = False
        self.comm = comm
        self.mesh_type = "firedrake_mesh"
        if abc_pad is None:
            self.abc_pad = 0.0
        elif abc_pad >= 0.0:
            self.abc_pad = abc_pad
        else:
            raise ValueError("abc_pad must be positive")

    def set_mesh_size(self, length_z=None, length_x=None, length_y=None):
        """
        Parameters
        ----------
        length_z : float, optional
            Length of the domain in the z direction. The default is None.
        length_x : float, optional
            Length of the domain in the x direction. The default is None.
        length_y : float, optional
            Length of the domain in the y direction. The default is None.

        Returns
        -------
        None
        """
        if length_z is not None:
            self.length_z = length_z
        if length_x is not None:
            self.length_x = length_x
        if length_y is not None:
            self.length_y = length_y

    def set_meshing_parameters(self, dx=None, cell_type=None, mesh_type=None):
        """
        Parameters
        ----------
        dx : float, optional
            Mesh size. The default is None.
        cell_type : str, optional
            Type of the cell. The default is None.
        mesh_type : str, optional
            Type of the mesh. The default is None.

        Returns
        -------
        None
        """
        if cell_type is not None:
            self.cell_type = cell_type
        if self.cell_type == "quadrilateral":
            self.quadrilateral = True
        if dx is not None:
            self.dx = dx
        if mesh_type is not None:
            self.mesh_type = mesh_type

    def make_periodic(self):
        """
        Sets the mesh boundaries periodic.
        """
        self.periodic = True
        if self.mesh_type != "firedrake_mesh":
            raise ValueError(
                "periodic mesh is only supported for firedrake_mesh"
            )

    def create_mesh(self):
        """
        Creates the mesh.

        Returns
        -------
        mesh : Firedrake Mesh
            Mesh
        """
        if self.dx is None and self.mesh_type == "firedrake_mesh":
            raise ValueError("dx is not set")
        elif self.mesh_type == "firedrake_mesh" and self.dimension == 2:
            return self.create_firedrake_2D_mesh()
        elif self.mesh_type == "firedrake_mesh" and self.dimension == 3:
            return self.create_firedrake_3D_mesh()
        else:
            raise ValueError("mesh_type is not supported")

    def create_firedrake_2D_mesh(self):
        """
        Creates a 2D mesh based on Firedrake meshing utilities.
        """
        nx = int(self.length_x / self.dx)
        nz = int(self.length_z / self.dx)
        comm = self.comm
        if self.cell_type == "quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False

        if self.periodic:
            return PeriodicRectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )
        else:
            return RectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )

    def create_firedrake_3D_mesh(self):
        dx = self.dx
        nx = int(self.length_x / dx)
        nz = int(self.length_z / dx)
        ny = int(self.length_y / dx)
        if self.cell_type == "quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False

        return BoxMesh(
            nz,
            nx,
            ny,
            self.length_z,
            self.length_x,
            self.length_y,
            quadrilateral=quadrilateral,
        )


# def create_firedrake_3D_mesh_based_on_parameters(dx, cell_type):
#     nx = int(self.length_x / dx)
#     nz = int(self.length_z / dx)
#     ny = int(self.length_y / dx)
#     if self.cell_type == "quadrilateral":
#         quadrilateral = True
#     else:
#         quadrilateral = False

#     return spyro.BoxMesh(
#         nz,
#         nx,
#         ny,
#         self.length_z,
#         self.length_x,
#         self.length_y,
#         quadrilateral=quadrilateral,
#     )


def RectangleMesh(nx, ny, Lx, Ly, pad=None, comm=None, quadrilateral=False):
    """Create a rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

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
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
    else:
        pad = 0
    mesh = fire.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def PeriodicRectangleMesh(
    nx, ny, Lx, Ly, pad=None, comm=None, quadrilateral=False
):
    """Create a periodic rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

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
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
    else:
        pad = 0
    mesh = fire.PeriodicRectangleMesh(
        nx, ny, Lx, Ly, quadrilateral=quadrilateral, comm=comm
    )
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def BoxMesh(nx, ny, nz, Lx, Ly, Lz, pad=None, quadrilateral=False):
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
        Lz += 2 * pad
    else:
        pad = 0
    quad_mesh = fire.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
    quad_mesh.coordinates.dat.data[:, 0] *= -1.0
    quad_mesh.coordinates.dat.data[:, 1] -= pad
    layer_height = Lz / nz
    mesh = fire.ExtrudedMesh(quad_mesh, nz, layer_height=layer_height)

    return mesh
