from firedrake import *


def advect(mesh, q, u, number_of_timesteps=10, output=False):
    """Advect a mesh with an indicator function based on the shape gradient `theta`
    solves a transport equation for `number_of_timesteps` using an upwinding DG scheme
    explictly marching in time with a 4th order RK scheme.
    """

    V = FunctionSpace(mesh, "DG", 0)

    dt = 0.0001
    T = dt * number_of_timesteps
    dtc = Constant(dt)
    q_in = Constant(1.0)

    dq_trial = TrialFunction(V)
    phi = TestFunction(V)
    a = phi * dq_trial * dx

    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    L1 = dtc * (
        q * div(phi * u) * dx
        - conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds
        - conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds
        - (phi("+") - phi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS
    )

    q1 = Function(V)
    q2 = Function(V)
    L2 = replace(L1, {q: q1})
    L3 = replace(L1, {q: q2})

    dq = Function(V)

    params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = LinearVariationalProblem(a, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = LinearVariationalProblem(a, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

    t = 0.0

    if output:
        indicator = File("indicator.pvd")

    step = 0
    while t < T - 0.5 * dt:

        solv1.solve()
        q1.assign(q + dq)

        solv2.solve()
        q2.assign(0.75 * q + 0.25 * (q1 + dq))

        solv3.solve()
        q.assign((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dq))

        if step % 5 == 0 and output:
            indicator.write(q)

        step += 1
        t += dt

    return q
