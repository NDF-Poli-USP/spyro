from firedrake import *
import sys

n = 20
mesh = RectangleMesh(n, n, 1, 1, diagonal="crossed")
#mesh = RectangleMesh(200, 200, 1, 1, diagonal="crossed") # plot the analytical solution

V = FunctionSpace(mesh, "CG", 2)
G = VectorFunctionSpace(mesh,"CG", 1)

num_timesteps=20
dt = 1.0/num_timesteps
coord_space = mesh.coordinates.function_space()
v = TestFunction(V)
u = TrialFunction(V)
sol = Function(V)
rho = Function(V)
xm = Function(mesh.coordinates, name="Physical coordinates")
xi = Function(mesh.coordinates, name="Computational coordinates")

uvec = TrialFunction(coord_space)
vvec = TestFunction(coord_space)
vel = Function(coord_space)

# interpolation of the analytical solution onto the initial mesh
#x,y = SpatialCoordinate(mesh)
#rho = Function(V).interpolate( 1. + 1 * tanh(100 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2))) )

#den = lambda t: 1 + t * tanh(100 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2)))
den = lambda t, x, y: 1 + t * 5 * exp(-50 * abs( (x-0.5)**2 + (y-0.5)**2 - (0.25)**2 ) ) # from cao etal 2002

if False:
    File("rho_refined.pvd").write(rho(1))
    sys.exit("exit")

t = 0.0

outfile_sol = File("./sol.pvd")
outfile_rho = File("./rho.pvd")

dx = Measure('dx', mesh)

for i in range(num_timesteps):
    print("Time = "+str(t))
    x,y = SpatialCoordinate(mesh)
    rho_t = den(t,x,y) / assemble(den(t,x,y) * dx)
    rho_tp1 = den(t+dt,x,y) / assemble(den(t+dt,x,y) * dx)
    a = inner( rho_t * grad(u), grad(v)) * dx
    L = inner((rho_tp1-rho_t)/dt, v) * dx
    bcs = DirichletBC(V, 0, (1,2,3,4))
    problem = LinearVariationalProblem(a, L, sol, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    #mesh.coordinates.assign(xi)
    solver.solve()
    
    if True:
        outfile_sol.write(sol, time=i)
        rho.interpolate(rho_t)
        outfile_rho.write(rho, time=i)
    
    # compute the velocity
    #vel = Function(G).interpolate( grad(sol) )
    a = inner(uvec, vvec) * dx
    L = inner(grad(sol), vvec) * dx
    bcs = DirichletBC(coord_space, 0, (1,2,3,4))
    problem = LinearVariationalProblem(a, L, vel, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solver.solve()

    # move the vertices
    xm.dat.data_with_halos[:,0] += vel.dat.data_with_halos[:,0] * dt 
    xm.dat.data_with_halos[:,1] += vel.dat.data_with_halos[:,1] * dt
    mesh.coordinates.assign(xm)
    
    t += dt
    

sys.exit("Exit")

