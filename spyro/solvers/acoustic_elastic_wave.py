import firedrake as fire
from firedrake import ds, dx, Constant, dot, grad
from acoustic_solver_construction_no_pml import construct_solver_or_matrix_no_pml

def acoustic_elastic_solver(Wave_object):
    # mesh
    mesh = Mesh("acoustic_elastic.msh")
    fluid_id     = 1
    solid_id     = 2
    interface_id = 3

    # ===============================================================================
    # function spaces:

    V_F = FunctionSpace(mesh, "CG", 2)
    V_S = VectorFunctionSpace(mesh, "CG", 2)

    # ===============================================================================
    # functions:

    # fluid
    p       = Function(V_F, name="p")
    p_n     = Function(V_F)
    p_nm1   = Function(V_F)
    p_trial = TrialFunction(V_F)
    q       = TestFunction(V_F)

    # solid
    u       = Function(V_S, name="u")
    u_n     = Function(V_S)
    u_nm1   = Function(V_S)
    u_trial = TrialFunction(V_S)
    v       = TestFunction(V_S)

    # ===============================================================================
    # normal unit vector:

    n   = FacetNormal(mesh)
    n_f = n("+")
    n_s = n("-")

    # ===============================================================================
    # solvers of equations:

    def eps(x):
        return sym(grad(x))

    def sigma(x):
        return lam * div(x) * Identity(mesh.geometric_dimension()) + 2 * mu * eps(x)

    # fluid
    ddot_p = (p_trial - 2.0*p_n + p_nm1) / Constant(dt * dt)

    F_p = (1/K) * ddot_p * q * dx(domain=mesh)\
          + q("+") * dot(sigma(u_n("-")) * n_f, n_f) * dS(interface_id) \
          + (1/rho_f) * dot(grad(q), grad(p_n)) * dx(fluid_id) \
          - (f/K) * q * dx(fluid_id)

    a_p, r_p = lhs(F_p), rhs(F_p)
    A_p = assemble(a_p)
    solver_f = LinearSolver(A_p, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    R_f = Cofunction(V_F.dual())

    # solid


    ddot_u = (u_trial - 2.0*u_n + u_nm1) / Constant(dt * dt)

    F_u = rho_s * dot(v, ddot_u) * dx(domain=mesh) \
          - p_n("+") * dot(v("-"), n_s) * dS(interface_id) \
          + inner(eps(v), sigma(u_n)) * dx(solid_id)

    a_u, r_u = lhs(F_u), rhs(F_u)
    A_u = assemble(a_u)
    solver_s = LinearSolver(A_u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    R_s = Cofunction(V_S.dual())

    # ===============================================================================

