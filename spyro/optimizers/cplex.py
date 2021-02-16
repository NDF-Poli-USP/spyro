from firedrake import *
import numpy as np
import cplex

def update_flip_limits(beta, counter, multiple, change, xi, mode='counter'):
    """update limit no number of 'flips'"""

    if mode == 'counter':
        condition = counter % multiple == 0
    elif mode == 'change':
        condition = change > 0

    if condition and counter > 0:
        new_beta = 0.9 * beta
        beta = np.maximum(new_beta, 1 / xi.size)
    
    return beta

def update_rmin(rmin, counter, limit, multiple):
    """update radius for sensitivity filer"""
    
    if counter >= limit and counter % multiple == 0:
        new_rmin = rmin / 2
        rmin = np.maximum(new_rmin, 1e-3)

    return rmin

def iterate_cplex(dJ, beta, xi):
    """Solve subproblem by Integer Linear Programming"""
    
    # Flip limits
    c1 =  (xi == 0).astype(int)
    c2 = -(xi == 1).astype(int)
    truncation = c1 + c2
    rhs = [int(beta * xi.size)]

    # Variable limits
    lb = 0 - xi
    ub = 1 - xi

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

    return xin

def optimize_cplex(dJ, beta, xi):
    """Solve optimization problem by Integer Linear Programming"""

    # Initial flip
    xi = iterate_cplex(dJ, beta, xi)

    return xi
