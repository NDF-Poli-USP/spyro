from firedrake import *
import numpy as np
import ipdb


class Dir_point_bc(DirichletBC):
    '''
    Class for Eikonal boundary conditions at a point.

    Attributes
    ----------
    nodes : `array`
        Points where the boundary condition is to be applied
    '''

    def __init__(self, V, value, nodes):
        '''
        Initialize the Dir_point_bc class.

        Parameters
        ----------
        V : `firedrake function space`
            Function space where the boundary condition is applied
        value : `firedrake constant`
            Value of the boundary condition
        nodes : `array`
            Points where the boundary condition is to be applied

        Returns
        -------
        None
        '''

        # Calling superclass init and providing a dummy subdomain id
        super(Dir_point_bc, self).__init__(V, value, 0)

        # Overriding the "nodes" property
        self.nodes = nodes


length_z = 1.0
length_x = 1.0
pad = 1.0

mesh_size = 0.05

nz = int(length_z / mesh_size)
nx = int(length_x / mesh_size)
npz = int((length_z + pad) / mesh_size)
npx = int((length_x + 2 * pad) / mesh_size)


# Create mesh
mesh_orig = RectangleMesh(nz, nx, length_z, length_x)
mesh_exte = RectangleMesh(npz, npx, length_z + pad, length_x + 2 * pad)

# Shift mesh to proper origin
mesh_exte.coordinates.dat.data[:, 0] -= pad
mesh_exte.coordinates.dat.data[:, 1] -= pad

# Define function space
Vo = FunctionSpace(mesh_orig, "CG", 1)
Ve = FunctionSpace(mesh_exte, "CG", 1)

# Original domain coordinates
z, x = SpatialCoordinate(mesh_orig)

# Create velocity function
cond_c = conditional(x < 0.5, 3.0, 1.5)
c_orig = Function(Vo, name="c_orig [km/s]")
c_orig.interpolate(cond_c)

# Initialize velocity in extende domain
c_exte = Function(Ve, name="c_exte [km/s]")
c_exte.interpolate(c_orig, allow_missing_dofs=True)

# Dirichlet BC on original domain boundary
bc_original = []
bc_nodes = [[], []]
for node, (zp, xp) in enumerate(mesh_exte.coordinates.dat.data[:]):
    if (0 <= zp <= length_z) and (0 <= xp <= length_x):
        val = c_exte.at(zp, xp)
        ids = np.atleast_1d(np.asarray(node))
        bc_original.append(Dir_point_bc(Ve, Constant(val), ids))
        bc_nodes[0].append(node)
        bc_nodes[1].append(val)

# Boundary nodes
bnd_ext = DirichletBC(Ve, 0., "on_boundary").nodes
for node in bnd_ext:
    z_bnd = np.clip(mesh_exte.coordinates.dat.data[node, 0], 0., length_z)
    x_bnd = np.clip(mesh_exte.coordinates.dat.data[node, 1], 0., length_x)
    if not ((0 < z_bnd < length_z) and (0 < x_bnd < length_x)):
        val = c_exte.at(z_bnd, x_bnd)
        ids = np.atleast_1d(np.asarray(node))
        bc_original.append(Dir_point_bc(Ve, Constant(val), ids))
        bc_nodes[0].append(node)
        bc_nodes[1].append(val)

# Create BC function
bc_func = Function(Ve)
bc_func.dat.data[bc_nodes[0]] = bc_nodes[1]

# Define variational problem
c = Function(Ve)
u = TrialFunction(Ve)
v = TestFunction(Ve)

# F = inner(grad(u), grad(v)) * dx - bc_func * v * dx
# F = inner(grad(u), grad(v)) * dx - div(grad(u - bc_func)) * v * dx
# F = inner(grad(u), grad(v)) * dx -\
#     (u - bc_func) * v * dx - div(grad(u - bc_func)) * v * dx
# F = inner(grad(u), grad(v)) * dx -\
#     (u - bc_func) * v * dx - div(grad(u)) * v * dx
# F = inner(grad(u), grad(v)) * dx - (u - bc_func) * v * dx
# F = inner(grad(u), grad(v)) * dx - u * v * dx
F = inner(grad(u), grad(v)) * dx

# Solve the problem
c.assign(0.)
solve(lhs(F) == rhs(F), c, bcs=bc_original,
      solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})


# For visualization
VTKFile("extended_velocity.pvd").write(c)

# alpha = 100.0
# ze, xe = SpatialCoordinate(mesh_exte)
# sol = 0.5 * c_exte * (tanh(alpha*(ze - length_z)) + 1) * \
#     (1 - 0.5 * (tanh(alpha * xe) + 1)) + \
#     0.5 * c_exte * (tanh(alpha*(xe - length_x) + 1))
# c = project(sol, Ve)
