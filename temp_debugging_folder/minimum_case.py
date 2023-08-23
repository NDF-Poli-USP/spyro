from firedrake import *
import numpy as np
set_log_level(30)


def make_mask(mask, sources, radius):
    """Make a mask for the domain."""
    x, y = SpatialCoordinate(mask.ufl_domain())
    for source in sources:
        x0, y0 = source
        mask = mask.interpolate(conditional( sqrt( (x - x0)**2 + (y - y0)**2 ) < radius, 1.0, mask))
    return mask

Lx = 1
Ly = 1
mesh = RectangleMesh(60, 60, Lx, Ly, diagonal="crossed")
mesh.coordinates.dat.data[:, 0] *= -1.0
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, 'CG', 1)

p = Function(V)
v = TestFunction(V)
u = TrialFunction(V)

mask = Function(V)
sources=[(-0.5, 0.25)]
radius=0.008
mask = make_mask(mask, sources, radius)
File('min_mask.pvd').write(mask)

k = Constant(1e6)
u0 = Constant(0.)

print('-------------------------------------------')
print('Solve Pre-Eikonal')
print('-------------------------------------------')

cond = conditional(y < 0.5, conditional(y < 0.25, 6.0, 3.0), 1.5)
c = Function(V).interpolate(cond)

f = Constant(1.0)
F1 = inner(grad(u), grad(v))*dx - f/c*v*dx  + mask * k * inner(u - u0, v) * dx

A = assemble(lhs(F1))

B = Function(V)
B = assemble(rhs(F1), tensor=B)

preeikonal = Function(V)
solve(A, preeikonal, B)
# zeros = np.where(preeikonal.dat.data[:] < 0.23)
# preeikonal.interpolate(conditional(preeikonal < 0.23, 0.0, preeikonal))
output = File('min_linear.pvd')
output.write(preeikonal)



print('-------------------------------------------')
print('Solved pre-eikonal')
print('-------------------------------------------')

f = Constant(1.0)
eps = k*CellDiameter(mesh)  # Stabilizer

weak_bc = mask * k * inner(preeikonal - u0, v) * dx

F = inner(sqrt(inner(grad(preeikonal), grad(preeikonal))),v) *dx + eps*inner(grad(preeikonal), grad(v))*dx - f / c*v*dx + weak_bc
L = 0

print('-------------------------------------------')
print('Solve eikonal')
print('-------------------------------------------')

solver_parameters = {'snes_type': 'vinewtonssls',
                    "snes_max_it": 1000,
                    "snes_atol": 5e-6,
                    "snes_rtol": 1e-20,
                    'snes_linesearch_type': 'l2',  # basic bt nleqerr cp l2
                    "snes_linesearch_damping": 1.0, # for basic,l2,cp
                    "snes_linesearch_maxstep": 0.5,  # bt,l2,cp
                    'snes_linesearch_order': 2,  # for newtonls com bt
                    'ksp_type': 'gmres',
                    'pc_type': 'lu',
                    'nonlinear_solver': 'snes',
                    'snes_monitor': None,
                    }

# bcs = DirichletBC(V, 0.0, (3))
solve(F == L, preeikonal, solver_parameters=solver_parameters)#{"newton_solver": {"relative_tolerance": 1e-6}})
output = File('min_nonlinear.pvd')
output.write(preeikonal)

# weak_bc = mask * k * inner(preeikonal - u0, v) * dx

# F = inner(sqrt(inner(grad(preeikonal), grad(preeikonal))),v) *dx + eps*inner(grad(preeikonal), grad(v))*dx - f / c*v*dx + weak_bc
# L = 0


print('-------------------------------------------')
print('Solved eikonal')
print('-------------------------------------------')