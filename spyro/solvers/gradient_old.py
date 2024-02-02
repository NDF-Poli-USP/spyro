# import firedrake as fire
# from firedrake import dx, ds, Constant, grad, inner, dot
# from firedrake.assemble import create_assembly_callable

# from ..domains import quadrature, space
# from ..pml import damping
# from ..io import ensemble_gradient
# from . import helpers

# # Note this turns off non-fatal warnings
# # set_log_level(ERROR)

# __all__ = ["gradient"]


# def gauss_lobatto_legendre_line_rule(degree):
#     fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
#     fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
#     finat_ps = finat.point_set.GaussLobattoLegendrePointSet
#     finat_qr = finat.quadrature.QuadratureRule
#     return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())


# # 3D
# def gauss_lobatto_legendre_cube_rule(dimension, degree):
#     make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
#     result = gauss_lobatto_legendre_line_rule(degree)
#     for _ in range(1, dimension):
#         line_rule = gauss_lobatto_legendre_line_rule(degree)
#         result = make_tensor_rule([result, line_rule])
#     return result


# @ensemble_gradient
# def gradient(
#     model,
#     mesh,
#     comm,
#     c,
#     receivers,
#     guess,
#     residual,
#     output=False,
#     save_adjoint=False,
# ):
#     """Discrete adjoint with secord-order in time fully-explicit timestepping scheme
#     with implementation of a Perfectly Matched Layer (PML) using
#     CG FEM with or without higher order mass lumping (KMV type elements).

#     Parameters
#     ----------
#     model: Python `dictionary`
#         Contains model options and parameters
#     mesh: Firedrake.mesh object
#         The 2D/3D triangular mesh
#     comm: Firedrake.ensemble_communicator
#         The MPI communicator for parallelism
#        c: Firedrake.Function
#         The velocity model interpolated onto the mesh nodes.
#     receivers: A :class:`spyro.Receivers` object.
#         Contains the receiver locations and sparse interpolation methods.
#     guess: A list of Firedrake functions
#         Contains the forward wavefield at a set of timesteps
#     residual: array-like [timesteps][receivers]
#         The difference between the observed and modeled data at
#         the receivers
#     output: boolean
#         optional, write the adjoint to disk (only for debugging)
#     save_adjoint: A list of Firedrake functions
#         Contains the adjoint at all timesteps

#     Returns
#     -------
#     dJdc_local: A Firedrake.Function containing the gradient of
#                 the functional w.r.t. `c`
#     adjoint: Optional, a list of Firedrake functions containing the adjoint

#     """

#     method = model["opts"]["method"]
#     degree = model["opts"]["degree"]
#     dimension = model["opts"]["dimension"]
#     dt = model["timeaxis"]["dt"]
#     tf = model["timeaxis"]["tf"]
#     nspool = model["timeaxis"]["nspool"]
#     fspool = model["timeaxis"]["fspool"]

#     params = {"ksp_type": "cg", "pc_type": "jacobi"}

#     element = fire.FiniteElement(
#         method, mesh.ufl_cell(), degree=degree, variant="spectral"
#     )

#     V = fire.FunctionSpace(mesh, element)

#     qr_x = gauss_lobatto_legendre_cube_rule(dimension=dimension, degree=degree)
#     qr_s = gauss_lobatto_legendre_cube_rule(
#         dimension=(dimension - 1), degree=degree
#     )

#     nt = int(tf / dt)  # number of timesteps

#     dJ = fire.Function(V, name="gradient")

#     # typical CG in N-d
#     u = fire.TrialFunction(V)
#     v = fire.TestFunction(V)

#     u_nm1 = fire.Function(V)
#     u_n = fire.Function(V)
#     u_np1 = fire.Function(V)

#     if output:
#         outfile = helpers.create_output_file("adjoint.pvd", comm, 0)

#     t = 0.0

#     # -------------------------------------------------------
#     m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=qr_x)
#     a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=qr_x)  # explicit

#     lhs1 = m1
#     rhs1 = -a

#     X = fire.Function(V)
#     B = fire.Function(V)

#     A = fire.assemble(lhs1, mat_type="matfree")
#     solver = fire.LinearSolver(A, solver_parameters=params)

#     # Define gradient problem
#     m_u = fire.TrialFunction(V)
#     m_v = fire.TestFunction(V)
#     mgrad = m_u * m_v * dx(rule=qr_x)

#     uuadj = fire.Function(V)  # auxiliarly function for the gradient compt.
#     uufor = fire.Function(V)  # auxiliarly function for the gradient compt.

#     ffG = 2.0 * c * dot(grad(uuadj), grad(uufor)) * m_v * dx(scheme=qr_x)

#     lhsG = mgrad
#     rhsG = ffG

#     gradi = fire.Function(V)
#     grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)

#     if method == "KMV":
#         grad_solver = fire.LinearVariationalSolver(
#             grad_prob,
#             solver_parameters={
#                 "ksp_type": "preonly",
#                 "pc_type": "jacobi",
#                 "mat_type": "matfree",
#             },
#         )
#     elif method == "CG":
#         grad_solver = fire.LinearVariationalSolver(
#             grad_prob,
#             solver_parameters={
#                 "mat_type": "matfree",
#             },
#         )

#     assembly_callable = create_assembly_callable(rhs1, tensor=B)

#     rhs_forcing = fire.Function(V)  # forcing term
#     if save_adjoint:
#         adjoint = [
#             fire.Function(V, name="adjoint_pressure") for t in range(nt)
#         ]
#     for step in range(nt - 1, -1, -1):
#         t = step * float(dt)
#         rhs_forcing.assign(0.0)
#         # Solver - main equation - (I)
#         # B = assemble(rhs_, tensor=B)
#         assembly_callable()

#         f = receivers.apply_receivers_as_source(rhs_forcing, residual, step)
#         # add forcing term to solve scalar pressure
#         B0 = B.sub(0)
#         B0 += f

#         # AX=B --> solve for X = B/Aˆ-1
#         solver.solve(X, B)

#         u_np1.assign(X)

#         # only compute for snaps that were saved
#         if step % fspool == 0:
#             # compute the gradient increment
#             uuadj.assign(u_np1)
#             uufor.assign(guess.pop())

#             grad_solver.solve()
#             dJ += gradi

#         u_nm1.assign(u_n)
#         u_n.assign(u_np1)

#         if step % nspool == 0:
#             if output:
#                 outfile.write(u_n, time=t)
#             if save_adjoint:
#                 adjoint.append(u_n)
#             helpers.display_progress(comm, t)

#     if save_adjoint:
#         return dJ, adjoint
#     else:
#         return dJ
