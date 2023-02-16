# test gradient of an elastic problem in a static setup

from firedrake import *
import finat
import math
import firedrake_adjoint
import sys
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL

print("Setting model definition")
mesh = RectangleMesh(30, 30, 1.5, 1.5, diagonal="crossed")

po = 3
V = VectorFunctionSpace(mesh, "Lagrange", po)
P = VectorFunctionSpace(mesh, "Lagrange", po, dim=2) # for lambda and mu inside PyROL
H = FunctionSpace(mesh, "Lagrange", po)

I = Identity(mesh.geometric_dimension())

# https://www.ndt.net/article/ndtce03/papers/v007/v007.htm
# concrete with fibers
rho = 2421.5 # kg/m3
g = 9.81 # m/s2
mu     = 24.62*1.e9 # Pa
lamb   = 14.04*1.e9 # Pa
mu_g   = 0.5 * mu # Pa
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

J_type=1 # 1) J is computed over the whole domain, 
         # 2) J is computed at (4, 0.25)

#bcs = DirichletBC(V, Constant((0.0, 0.0)), (1,2,3,4)) # all faces are clamped
#bcs = DirichletBC(V, Constant((0.0, 0.0)), (1)) # face 1 is clamped
#bcs = DirichletBC(V, Constant((0.0, 0.0)), (3)) # face 3 is clamped
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
if False:
    mu_guess = Function(H).interpolate(Constant(mu)) # Pa
    lamb_guess = Function(H).interpolate(Constant(lamb_g)) # Pa
    f_guess = exf 
    bcs_guess = bcs 
    s_guess = solve_elastic_problem(lamb_guess, mu_guess, f_guess, bcs_guess)
    if False:
        s_guess.rename('displacement')
        outfile2 = File("displacement_guess.pvd")
        outfile2.write(s_guess)
#}}}
print("Computing functional (exact - guess)") #{{{
if False:
    if J_type==1:
        J = J_scale * assemble( (0.5 * inner(s_guess-s_exact, s_guess-s_exact)) * dx) # == s_exact-s_guess
    elif J_type==2:
        rec_position = [(4, 0.25)]
        point_cloud = VertexOnlyMesh(mesh, rec_position)
        P = VectorFunctionSpace(point_cloud, "DG", 0)
        misfit_at_rec = interpolate(s_exact-s_guess, P)
        J = J_scale * assemble(0.5 * inner(misfit_at_rec, misfit_at_rec) * dx)
    else:
        J = 0
#}}}
print("Computing gradient - AD module") #{{{
if False:
    control_mu = firedrake_adjoint.Control(mu_guess)
    control_lamb = firedrake_adjoint.Control(lamb_guess)
    #dJdm_AD  = firedrake_adjoint.compute_gradient(J, control_mu)
    #dJdl_AD  = firedrake_adjoint.compute_gradient(J, control_lamb)
    dJdm_AD  = firedrake_adjoint.compute_gradient(J, control_mu, options={"riesz_representation": "L2"})
    dJdl_AD  = firedrake_adjoint.compute_gradient(J, control_lamb, options={"riesz_representation": "L2"})
    #Note that we set the “riesz_representation” option to “L2” in compute_gradient(). It indicates that the gradient should not be returned as an operator, that is not in the dual space 𝑉∗, but instead its Riesz representation in the primal space 𝑉. This is necessary to plot the sensitivities without seeing mesh dependent features.
    #http://www.dolfin-adjoint.org/en/latest/documentation/klein/klein.html
    File("dJdl_AD.pvd").write(dJdl_AD)
    File("dJdm_AD.pvd").write(dJdm_AD)
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
if J_type==2:
   sys.exit("exiting without computing gradient by adjoint method") 
print("Computing gradient - adjoint method") #{{{
if False:
    mu_adjoint = Function(H).interpolate(Constant(mu)) # Pa
    lamb_adjoint = Function(H).interpolate(Constant(lamb_g)) # Pa
    f_adjoint = J_scale * (s_exact-s_guess) # this is the correct sign
    bcs_adjoint = bcs # ISSM employs the same Dirichlet BC on both forward and adjoint models
    s_adjoint = solve_elastic_problem(lamb_adjoint, mu_adjoint, f_adjoint, bcs_adjoint)
    if False:
        s_adjoint.rename('adjoint')
        outfile1 = File("adjoint.pvd")
        outfile1.write(s_adjoint)

    dJdl_adj, dJdm_adj = compute_gradient_via_adjoint(s_guess, s_adjoint)
    File("dJdl_adj.pvd").write(dJdl_adj)
    File("dJdm_adj.pvd").write(dJdm_adj)
#}}}

if False:
    delta_dJdl = Function(H).assign(dJdl_adj-dJdl_AD)
    delta_dJdm = Function(H).assign(dJdm_adj-dJdm_AD)
    File("delta_dJdl.pvd").write(delta_dJdl)
    File("delta_dJdm.pvd").write(delta_dJdm)

print("Starting inversion run") 
class L2Inner(object): #{{{
    def __init__(self):
        self.A = assemble( TrialFunction(H) * TestFunction(H) * dx, mat_type="matfree")
        self.Ap = as_backend_type(self.A).mat()
        self.ulp = Function(H)
        self.ump = Function(H)
        self.vlp = Function(H)
        self.vmp = Function(H)

    def eval(self, u, v):
        self.ulp.dat.data[:] = u.dat.data[:,0] # lambda
        self.vlp.dat.data[:] = v.dat.data[:,0] # lambda
        self.ump.dat.data[:] = u.dat.data[:,1] # mu
        self.vmp.dat.data[:] = v.dat.data[:,1] # mu
        ulpet = as_backend_type(self.ulp.vector()).vec() # lambda
        vlpet = as_backend_type(self.vlp.vector()).vec() # lambda
        umpet = as_backend_type(self.ump.vector()).vec() # mu
        vmpet = as_backend_type(self.vmp.vector()).vec() # mu
        A_ul = self.Ap.createVecLeft() # lambda
        A_um = self.Ap.createVecLeft() # mu
        self.Ap.mult(ulpet, A_ul) # lambda
        self.Ap.mult(umpet, A_um) # mu
        return vlpet.dot(A_ul) + vmpet.dot(A_um)
#}}}
class ObjectiveElastic(ROL.Objective): #{{{
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.s_guess = None # Z,X displacements
        self.misfit  = 0.0 
        self.lamb    = Function(H) 
        self.mu      = Function(H)  
        self.f       = f_exact
        self.bcs     = bcs_exact
        self.s_exact = s_exact 

    def value(self, x, tol):
        """Compute the functional"""
        self.s_guess = solve_elastic_problem(self.lamb, self.mu, self.f, self.bcs)
        self.misfit = J_scale * (self.s_exact - self.s_guess)
        J = J_scale * assemble( (0.5 * inner(self.s_guess-self.s_exact, self.s_guess-self.s_exact)) * dx) 
        return J

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        s_adjoint = solve_elastic_problem(self.lamb, self.mu, self.misfit, self.bcs) 
        
        dJdl_adj, dJdm_adj = compute_gradient_via_adjoint(self.s_guess, s_adjoint)
        File("dJdl_adj.pvd").write(dJdl_adj)
        File("dJdm_adj.pvd").write(dJdm_adj)
        
        g.scale(0) # it zeros the entire vector
        g.vec.dat.data[:,0] += dJdl_adj.dat.data[:]
        g.vec.dat.data[:,1] += dJdm_adj.dat.data[:]

    def update(self, x, flag, iteration):
        self.lamb.dat.data[:] = x.vec.dat.data[:,0]
        self.mu.dat.data[:] = x.vec.dat.data[:,1]


#}}}
# ROL definition and simulation {{{
paramsDict = {
    "General": {
        "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 25}
    },
    "Step": {
        "Type": "Augmented Lagrangian",
        "Line Search": {
            "Descent Method": {
                "Type": "Quasi-Newton Step"
            }
        },
        "Augmented Lagrangian": {
            'Initial Penalty Parameter'               : 1.e2,
            'Penalty Parameter Growth Factor'         : 2,
            'Minimum Penalty Parameter Reciprocal'    : 0.1,
            'Initial Optimality Tolerance'            : 1.0,
            'Optimality Tolerance Update Exponent'    : 1.0,
            'Optimality Tolerance Decrease Exponent'  : 1.0,
            'Initial Feasibility Tolerance'           : 1.0,
            'Feasibility Tolerance Update Exponent'   : 0.1,
            'Feasibility Tolerance Decrease Exponent' : 0.9,
            'Print Intermediate Optimization History' : True,
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 10,
        }, 
    },
    "Status Test": {
        "Gradient Tolerance": 1e-15,
        'Relative Gradient Tolerance': 1e-10,
        "Step Tolerance": 1.0e-15,
        'Relative Step Tolerance': 1e-15,
        "Iteration Limit": 60
    },
}

params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()
#inner_product = None

x0 = Function(P)
xlo = Function(P)
xup = Function(P)

x0.dat.data[:,0] = lamb_g # lambda
x0.dat.data[:,1] = mu_g # mu

xlo.dat.data[:,0] = 0.1 * lamb_g
xlo.dat.data[:,1] = 0.1 * mu_g

xup.dat.data[:,0] = 5.0 * lamb_g
xup.dat.data[:,1] = 5.0 * mu_g

opt  = FeVector(x0.vector(), inner_product)
x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)

obj  = ObjectiveElastic(inner_product)
algo = ROL.Algorithm("Line Search", params)
algo.run(opt, obj, bnd)

File("final_lamb_elastic.pvd").write(obj.lamb)
File("final_mu_elastic.pvd").write(obj.mu)

error_lamb = Function(H).assign( 100*(lamb_exact-obj.lamb)/lamb_exact )
error_mu   = Function(H).assign( 100*(mu_exact-obj.mu)/mu_exact )
File("error_lamb.pvd").write(error_lamb)
File("error_mu.pvd").write(error_mu)
#}}}
