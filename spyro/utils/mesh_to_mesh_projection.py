import firedrake
from firedrake.supermeshing import assemble_mixed_mass_matrix

def mesh_to_mesh_projection(source, target, solver_parameters=None, degree=None):
    """Performs a mesh-to-mesh projection using Galerkin projection and supermesh scheme.
    It overloads  Firedrake projection function to consider ML (KMV) elements."""
    
    if solver_parameters is None:
        solver_parameters = {}
    else:
        solver_parameters = solver_parameters.copy()
    
    # original from firedrake projection
    solver_parameters.setdefault("ksp_type", "cg")
    solver_parameters.setdefault("ksp_rtol", 1e-8)
    solver_parameters.setdefault("pc_type", "bjacobi")
    solver_parameters.setdefault("sub_pc_type", "icc")

    # create matrix A
    u = firedrake.TrialFunction(target.function_space())
    v = firedrake.TestFunction(target.function_space())
    a = firedrake.inner(u, v)*firedrake.dx(degree=degree) 
    A = firedrake.assemble(a, bcs=None,
                           mat_type=solver_parameters.get("mat_type"),
                           form_compiler_parameters=None)

    # create solver
    solver = firedrake.LinearSolver(A, solver_parameters=solver_parameters)
    
    # create rhs
    rhs = firedrake.Function(target.function_space())

    # assemble mixed mass matrix using supermeshing 
    mixed_mass = assemble_mixed_mass_matrix(source.function_space(), target.function_space())

    # setting rhs
    with source.dat.vec_ro as u, rhs.dat.vec_wo as v:
        mixed_mass.mult(u, v)

    # now, solve the linear system, projection the solution onto target
    solver.solve(target, rhs)
    
    return target
