from firedrake import *
import cplex

def iterate_cplex(dJ, beta, xi):
    """Solve subproblem by Integer Linear Programming"""
    
    # Flip limits
    c1 =  (xi == 0).astype(int)
    c2 = -(xi == 1).astype(int)
    truncation = c1 + c2
    rhs = [int(beta * xi.size)]

    # Variable limits
    lb = -1*(abs(xi - 1) < 0.001)
    ub =  1*(abs(xi) < 0.001)

    # Set up problem
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.variables.add(
        obj = dJ, lb = lb.tolist(), ub = ub.tolist(), types = "I"*xi.size)

    # Impose flip limits
    rows = [[[i for i in range(xi.size)], truncation.tolist()]]
    problem.linear_constraints.add(lin_expr=rows, senses="L", rhs=rhs)
    
    # Solve sub-problem
    problem.solve()

    # Update values
    dxi = np.array(problem.solution.get_values())
    xin = xi + dxi

    # Iteration info
    iter_info = "it.: {:d} | obj.f.: {:e} | rel.var.: {: 2.2f}% | move: {:g}"

    print(iter_info.format(0, J, 1, rhs[0]))

    return xin
