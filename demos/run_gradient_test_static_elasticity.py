# test gradient of an elastic problem in a static setup

from firedrake import *
import finat
import math
import firedrake_adjoint
import sys

print("Setting model definition")
mesh = RectangleMesh(50, 50, 1.5, 1.5)

po = 4
#V = VectorFunctionSpace(mesh, "Lagrange", po)
#H = FunctionSpace(mesh, "Lagrange", po)
V = VectorFunctionSpace(mesh, "Lagrange", po)
H = FunctionSpace(mesh, "Lagrange", po)

I = Identity(mesh.geometric_dimension())

# https://www.ndt.net/article/ndtce03/papers/v007/v007.htm
# concrete with fibers
rho = 2421.5 # kg/m3
g = 9.81 # m/s2
mu = 24.62*1.e9 # Pa
lamb = 14.04*1.e9 # Pa
mu_g = 0.5 * mu # Pa
lamb_g = 0.5 * lamb # Pa
dist_force = 10000000 # N/m

def D(w):   # strain tensor
    return 0.5 * (grad(w) + grad(w).T)

def sigma(w, lamb, mu, I): # elastic stress tensor
    return lamb * tr(D(w)) * I + 2 * mu * D(w) 

def solve_elastic_problem(lamb, mu, fext, bcs): # solve the elastic problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = lamb * tr(D(u)) * tr(D(v)) * dx + 2.0 * mu * inner(D(u),D(v)) * dx
    f = inner(fext,v) * dx
    s = Function(V) # solution
    par = {"ksp_type": "preonly", "pc_type": "lu"} # direct solver
    solve(a == f, s, bcs=bcs, solver_parameters=par)
    return s

def compute_gradient_via_adjoint(s_guess, s_adjoint):
    u = TrialFunction(H)
    v = TestFunction(H)
    
    m = u * v * dx
    al = tr(D(s_guess)) * tr(D(s_adjoint)) * v * dx # w.r.t. lambda
    am = 2.0 * inner(D(s_guess), D(s_adjoint)) * v * dx # w.r.t. mu

    dJdl = Function(H)
    dJdm = Function(H)

    par = {"ksp_type": "preonly", "pc_type": "lu"} # direct solver
    solve(m == al, dJdl, solver_parameters=par) 
    solve(m == am, dJdm, solver_parameters=par) 
    return dJdl, dJdm

#bcs = DirichletBC(V, Constant((0.0, 0.0)), (1,2,3,4)) # all faces are clamped
#bcs = DirichletBC(V, Constant((0.0, 0.0)), (1)) # face 1 is clamped
bcs = [ DirichletBC(V.sub(1), Constant(0), 3), DirichletBC(V.sub(0), Constant(0.0), (1,2)) ] # 1D case, vertical disp. 
exf = Constant((0.0, - rho * g - dist_force)) # adding an external force (distributed)

J_scale = 1.e20 # O(misfit) = 1e-10, O(lambda) = 1.e10, O(dJdl) = 1e-20 

print("Solving forward problem (exact)") #{{{
mu_exact = Function(H).interpolate(Constant(mu)) # Pa
lamb_exact = Function(H).interpolate(Constant(lamb)) # Pa
f_exact = exf 
bcs_exact = bcs
s_exact = solve_elastic_problem(lamb_exact, mu_exact, f_exact, bcs_exact)
if True:
    s_exact.rename('displacement')
    outfile1 = File("displacement_exact.pvd")
    outfile1.write(s_exact)
#}}}
print("Solving forward problem (guess)") #{{{
mu_guess = Function(H).interpolate(Constant(mu_g)) # Pa
lamb_guess = Function(H).interpolate(Constant(lamb_g)) # Pa
f_guess = exf 
bcs_guess = bcs 
s_guess      = solve_elastic_problem(lamb_guess, mu_guess, f_guess, bcs_guess)
s_guess_lamb = solve_elastic_problem(lamb_guess, mu_exact, f_guess, bcs_guess)
s_guess_mu   = solve_elastic_problem(lamb_exact, mu_guess, f_guess, bcs_guess)
if True:
    s_guess.rename('displacement')
    outfile2 = File("displacement_guess.pvd")
    outfile2.write(s_guess)
#}}}
print("Computing functional (exact - guess)") #{{{
J   = J_scale * assemble( (0.5 * inner(s_guess-s_exact, s_guess-s_exact)) * dx) # == s_exact-s_guess
J_l = J_scale * assemble( (0.5 * inner(s_guess_lamb-s_exact, s_guess_lamb-s_exact)) * dx) # == s_exact-s_guess
J_m = J_scale * assemble( (0.5 * inner(s_guess_mu-s_exact, s_guess_mu-s_exact)) * dx) # == s_exact-s_guess
#}}}
print("Computing gradient - AD module") #{{{
control_mu   = firedrake_adjoint.Control(mu_guess)
control_lamb = firedrake_adjoint.Control(lamb_guess)
dJdm_AD_m  = firedrake_adjoint.compute_gradient(J_m, control_mu, options={"riesz_representation": "L2"})
dJdl_AD_l  = firedrake_adjoint.compute_gradient(J_l, control_lamb, options={"riesz_representation": "L2"})
dJdm_AD    = firedrake_adjoint.compute_gradient(J, control_mu, options={"riesz_representation": "L2"})
dJdl_AD    = firedrake_adjoint.compute_gradient(J, control_lamb, options={"riesz_representation": "L2"})
File("dJdl_AD.pvd").write(dJdl_AD)
File("dJdm_AD.pvd").write(dJdm_AD)
#Note that we set the “riesz_representation” option to “L2” in compute_gradient(). It indicates that the gradient should not be returned as an operator, that is not in the dual space 𝑉∗, but instead its Riesz representation in the primal space 𝑉. This is necessary to plot the sensitivities without seeing mesh dependent features.
#http://www.dolfin-adjoint.org/en/latest/documentation/klein/klein.html
if False: #{{{
    print("Mass matrix post-processing")
    mu = TrialFunction(H)
    mv = TestFunction(H)
    fl = Function(H)
    fm = Function(H)
    dJdl_AD_mm = Function(H)
    dJdm_AD_mm = Function(H)

    mm = mu * mv * dx # mass matrix

    fl.assign(dJdl_AD)
    fm.assign(dJdm_AD)

    Fl = mm - fl * mv * dx
    Fm = mm - fm * mv * dx

    dJdl_prob = LinearVariationalProblem(lhs(Fl), rhs(Fl), dJdl_AD_mm)
    dJdm_prob = LinearVariationalProblem(lhs(Fm), rhs(Fm), dJdm_AD_mm)

    dJdl_solver = LinearVariationalSolver(dJdl_prob, solver_parameters=par)
    dJdm_solver = LinearVariationalSolver(dJdm_prob, solver_parameters=par)

    dJdl_solver.solve()
    dJdm_solver.solve()

    File("dJdl_AD_mm.pvd").write(dJdl_AD_mm)
    File("dJdm_AD_mm.pvd").write(dJdm_AD_mm)
#}}}
#}}}
print("Computing gradient - adjoint method") #{{{
s_adjoint      = solve_elastic_problem(lamb_guess, mu_guess, J_scale * (s_exact-s_guess), bcs_exact)
s_adjoint_mu   = solve_elastic_problem(lamb_exact, mu_guess, J_scale * (s_exact-s_guess_mu), bcs_exact)
s_adjoint_lamb = solve_elastic_problem(lamb_guess, mu_exact, J_scale * (s_exact-s_guess_lamb), bcs_exact)

if True:
    s_adjoint.rename('adjoint')
    outfile1 = File("adjoint.pvd")
    outfile1.write(s_adjoint)

dJdl_adj, dJdm_adj = compute_gradient_via_adjoint(s_guess, s_adjoint)
dJdl_adj_l, _ = compute_gradient_via_adjoint(s_guess_lamb, s_adjoint_lamb)
_, dJdm_adj_m = compute_gradient_via_adjoint(s_guess_mu, s_adjoint_mu)
File("dJdl_adj.pvd").write(dJdl_adj)
File("dJdm_adj.pvd").write(dJdm_adj)
#}}}
print("Computing the difference - adjoint minus AD") #{{{
delta_dJdl = Function(H).assign(dJdl_adj-dJdl_AD)
delta_dJdm = Function(H).assign(dJdm_adj-dJdm_AD)
File("delta_dJdl.pvd").write(delta_dJdl)
File("delta_dJdm.pvd").write(delta_dJdm)
#}}}

#sys.exit("exiting without computing gradient by finite difference method") 
print("Computing gradient - finite difference")
steps = [1e-3, 1e-4, 1e-5, 1e-6]  # step length

delta_mu = Function(H)
delta_lamb = Function(H)
delta_mu.assign(mu_guess)
delta_lamb.assign(lamb_guess) 

# this deepcopy is important otherwise pertubations accumulate
lamb_original = lamb_guess.copy(deepcopy=True)
mu_original = mu_guess.copy(deepcopy=True)

for step in steps:  # range(3):
    # perturb the model and calculate the functional (again)
    # J(m + delta_m*h)
    lamb_guess = lamb_original + step * delta_lamb
    mu_guess = mu_original + step * delta_mu

    s_guess_fd_lamb = solve_elastic_problem(lamb_guess, mu_exact, f_guess, bcs_guess)
    s_guess_fd_mu   = solve_elastic_problem(lamb_exact, mu_guess, f_guess, bcs_guess)
    
    Jp_l = J_scale * assemble( (0.5 * inner(s_guess_fd_lamb-s_exact, s_guess_fd_lamb-s_exact)) * dx) 
    Jp_m   = J_scale * assemble( (0.5 * inner(s_guess_fd_mu-s_exact, s_guess_fd_mu-s_exact)) * dx) 
    
    projnorm_dJdl_AD  = assemble(dJdl_AD  * delta_lamb * dx)
    projnorm_dJdl_adj = assemble(dJdl_adj * delta_lamb * dx)
    projnorm_dJdm_AD  = assemble(dJdm_AD  * delta_mu * dx)
    projnorm_dJdm_adj = assemble(dJdm_adj * delta_mu * dx)
    
    projnorm_dJdl_AD_l  = assemble(dJdl_AD_l  * delta_lamb * dx)
    projnorm_dJdl_adj_l = assemble(dJdl_adj_l * delta_lamb * dx)
    projnorm_dJdm_AD_m  = assemble(dJdm_AD_m  * delta_mu * dx)
    projnorm_dJdm_adj_m = assemble(dJdm_adj_m * delta_mu * dx)
    
    fd_grad_lamb = (Jp_l - J_l) / step
    fd_grad_mu   = (Jp_m - J_m) / step
    
    print(
        "\n Step " + str(step) + "\n" 
        + "\t lambda:\n"
        + "\t cost functional (exact):\t" + str(J_l) + "\n"
        + "\t cost functional (FD):\t\t" + str(Jp_l) + "\n"
        + "\t grad'*dir (adj):\t\t" + str(projnorm_dJdl_adj_l) + "\n"
        + "\t grad'*dir (AD):\t\t" + str(projnorm_dJdl_AD_l) + "\n"
        + "\t grad'*dir (FD):\t\t" + str(fd_grad_lamb) + "\n"
        + "\n"
        + "\t mu:\n"
        + "\t cost functional (exact):\t" + str(J_m) + "\n"
        + "\t cost functional (FD):\t\t" + str(Jp_m) + "\n"
        + "\t grad'*dir (adj):\t\t" + str(projnorm_dJdm_adj_m) + "\n"
        + "\t grad'*dir (AD):\t\t" + str(projnorm_dJdm_AD_m) + "\n"
        + "\t grad'*dir (FD):\t\t" + str(fd_grad_mu) + "\n"
    )



