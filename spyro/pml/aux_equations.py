
import firedrake as fire
from . import damping
def set_pml_aux_eq(V, sigma, u_n, v, FF, c, dt, qr_x, dim, params):

    Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    if dim == 2:
        pp      = fire.TrialFunction(Z)
        qq      = fire.TestFunction(Z)
        pp_np1  = fire.Function(Z)
        pp_n    = fire.Function(Z)
        pp_nm1  = fire.Function(Z)
        sigma_x = sigma[0]
        sigma_z = sigma[1]
        Gamma_1, Gamma_2 = damping.matrices_2D(
                                    sigma_z, sigma_x)

        pml2 = sigma_x * sigma_z * u_n * v * fire.dx(rule=qr_x)
        pml3 = fire.inner(pp_n, fire.grad(v)) * fire.dx(rule=qr_x)
        FF  += pml2 + pml3
        # -------------------------------------------------------
        mm1   = (fire.dot((pp - pp_n), qq) / fire.Constant(dt)) * fire.dx(rule=qr_x)
        mm2   = fire.inner(fire.dot(Gamma_1, pp_n), qq) * fire.dx(rule=qr_x)
        dd    = c * c * fire.inner(fire.grad(u_n), fire.dot(Gamma_2, qq)) * fire.dx(rule=qr_x)
        RR    = mm1 + mm2 + dd
        lhs__ = fire.lhs(RR)
        rhs__ = fire.rhs(RR)
        X0    = fire.Function(Z)
        
        problem0 = fire.LinearVariationalProblem(lhs__, rhs__, X0)
        solver0  = fire.LinearVariationalSolver(problem0, solver_parameters=params)            
        pp_v     = [pp_n, pp_nm1, pp_np1]
        return solver0, pp_v, X0

    elif dim == 3:     
        sigma_x = sigma[0]
        sigma_z = sigma[1]
        sigma_y = sigma[2]
        Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(
                                            sigma_x, sigma_y, sigma_z)
        pp      = fire.TrialFunction(Z)
        qq      = fire.TestFunction(Z)
        pp_np1  = fire.Function(Z)
        pp_n    = fire.Function(Z)
        pp_nm1  = fire.Function(Z)
        psi     = fire.TrialFunction(V)
        phi     = fire.TestFunction(V)
        psi_n   = fire.Function(V)
        psi_np1 = fire.Function(V)
        psi_nm1 = fire.Function(V)   


        pml2 = (
            (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
            * u_n
            * v
            * fire.dx(rule=qr_x)
        )
        pml3 = (sigma_x * sigma_y * sigma_z) * psi_n * v * fire.dx(rule=qr_x)
        pml4 = fire.inner(pp_n, fire.grad(v)) * fire.dx(rule=qr_x)

        FF += pml2 + pml3 + pml4

        # -------------------------------------------------------
        mm1 = (fire.dot((pp - pp_n), qq) / fire.Constant(dt)) * fire.dx(rule=qr_x)
        mm2 = fire.inner(fire.dot(Gamma_1, pp_n), qq) * fire.dx(rule=qr_x)
        dd1 = c * c * fire.inner(fire.grad(u_n), fire.dot(Gamma_2, qq)) * fire.dx(rule=qr_x)
        dd2 = -c * c * fire.inner(fire.grad(psi_n), fire.dot(Gamma_3, qq)) * fire.dx(rule=qr_x)
        RR  = mm1 + mm2 + dd1 + dd2
        
        lhs__ = fire.lhs(RR)
        rhs__ = fire.rhs(RR)
        X0    = fire.Function(Z)
  
        problem0 = fire.LinearVariationalProblem(lhs__, rhs__, X0)
        solver0  = fire.LinearVariationalSolver(problem0, solver_parameters=params)            

        # -------------------------------------------------------
        mmm1 = (fire.dot((psi - psi_n), phi) / fire.Constant(dt)) * fire.dx(rule=qr_x)
        uuu1 = (-u_n * phi) * fire.dx(rule=qr_x)
        QQ   = mmm1 + uuu1

        lhs___ = fire.lhs(QQ)
        rhs___ = fire.rhs(QQ)
        X1     = fire.Function(V)

        problem1 = fire.LinearVariationalProblem(lhs___, rhs___, X1)
        solver1  = fire.LinearVariationalSolver(problem1, solver_parameters=params)            
        pp_v     = [pp_n, pp_nm1, pp_np1]
        psi_v    = [psi_n, psi_nm1, psi_np1]
        
        return solver0, solver1, pp_v, psi_v, X1


