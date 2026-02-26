import firedrake as fire
from firedrake import dS, dx, Constant, dot, grad, inner, div, Identity
from acoustic_solver_construiction_no_pml import construct_solver_or_matrix_no_pml
from elastic_wave.forms import istropic_elastic_without_pml

def eps(x):
    return sym(grad(x))

def sigma(x):
    return lam * div(x) * Identity(mesh.geometric_dimension()) + 2 * mu * eps(x)

def construct_acoustic_elastic_solver_no_pml(AcousticWave, ElasticWave, interface_id):
     """
     It extends the basic formulation to include acoustic-elastic coupling.
     """

     n = fire.FacetNormal(AcousticWave.mesh)
     n_f = n("+")
     n_s = n("-")
     
     # Fluid:
     q = fire.TestFunction(AcousticWave.function_space)

     u_elastic = ElasticWave.u_n
     term_interface_fluid = q("+") * dot(sigma(u_elastic("-")) * n_f, n_f) * dS(interface_id)

     AcousticWave.rhs += term_interface_fluid
     A_f = fire.assemble(AcousticWave.lhs)
     AcousticWave.solver = fire.LinearSolver(A_f, solver_parameters=AcousticWave.solver_parameters)

     # Solid:
     v = fire.TestFunction(ElasticWave.function_space)
     p_fluid = AcousticWave.un

     term_interface_solid = - p_fluid("+") * dot(v("-"), n_s) * dS(interface_id)
     ElasticWave.rhs += term_interface_solid
     A_s = fire.assemble(ElasticWave.lhs)
     ElasticWave.solver = fire.LinearSolver(A_s, solver_parameters=ElasticWave.solver_parameters)
