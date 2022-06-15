# based on the mesh shown in https://www.firedrakeproject.org/demos/linear_fluid_structure_interaction.py.html
from firedrake import *
import numpy as np

fluid_id = 1  # fluid subdomain
structure_id = 2  # structure subdomain
bottom_id = 1  # structure bottom
top_id = 6  # fluid surface
#mesh = Mesh("./meshes/L_domain.msh")
mesh = Mesh("./meshes/L_domain_mod.msh") # contains tag=13 over two facets 

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v  = TestFunction(V)
f = Function(V).interpolate(Constant(1.))

a = inner(grad(u), grad(v)) * dx
L = f*v*dx

sol = Function(V, name="u")

#bc = DirichletBC(V, 0, [13]) # 5, 9, 10 for the lateral boundaries of the structure
x, y = mesh.coordinates
markers = Function(V).interpolate(conditional(x < 0, 0, 1))
File("markers.pvd").write(markers) # marker
class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

bc = MyBC(V, 0, markers)

solve(a == L, sol, bc)

File("u.pvd").write(sol) # u
