import firedrake
from firedrake.petsc import PETSc
from pyadjoint import no_annotations
import ufl
import numpy as np

r"""
The elliptic Monge-Ampere equation is solved using a quasi-Newton
method (see McRae et al. 2018 for details).

References
==========
A. T. T. McRae, C. J. Cotter, C. J. Budd, Optimal-transport-based
mesh adaptivity on the plane and sphere using finite elements, SIAM
Journal on Scientific Computing 40 (2) (2018) 1121–1148.
doi:10.1137/16M1109515.
"""

# solver parameters - from `movement', https://github.com/pyroteus/movement {{{
_serial_qn = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_ksp_max_it": 5,
    "fieldsplit_0_mg_levels_pc_type": "ilu",
    "fieldsplit_1_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "ksp_max_it": 200,
    "snes_max_it": 125,
    "ksp_gmres_restart": 200,
    "snes_rtol": 1.0e-08,
    "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 5,
    "snes_linesearch_maxstep": 1.05,
    "snes_linesearch_damping": 0.8,
    "snes_lag_preconditioner": -2,
}

_parallel_qn = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_ksp_max_it": 5,
    "fieldsplit_0_mg_levels_pc_type": "bjacobi",
    "fieldsplit_0_mg_levels_sub_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_sub_pc_type": "ilu",
    "fieldsplit_1_pc_type": "bjacobi",
    "fieldsplit_1_sub_ksp_type": "preonly",
    "fieldsplit_1_sub_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "ksp_max_it": 200,
    "snes_max_it": 125,
    "ksp_gmres_restart": 200,
    "snes_rtol": "1.0e-08",
    "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 5,
    "snes_linesearch_maxstep": 1.05,
    "snes_linesearch_damping": 0.8,
    "snes_lag_preconditioner": -2,
}

_mass_inv = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
}

_jacobi = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

_cg = {
    "ksp_type": "cg",
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu",
}

_lu = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
#}}}

def monge_ampere_solver(mesh, monitor_function, 
                        p = 2, 
                        rtol = 1.0e-03, 
                        maxiter = 1000, 
                        fix_boundary_nodes = False, 
                        **kwargs):
    
    # Only works for two-dimensional meshes
    dim = mesh.topological_dimension()
    if dim != 2:
        raise NotImplementedError(f"Dimension {dim} has not been considered yet")

    # Measures
    degree = kwargs.get('quadrature_degree')
    dx = firedrake.dx(domain=mesh, degree=degree)
    ds = firedrake.ds(domain=mesh, degree=degree)

    # Mesh coordinate functions
    x  = firedrake.Function(mesh.coordinates, name="Physical coordinates")
    xi = firedrake.Function(mesh.coordinates, name="Computational coordinates")
    
    # Create function spaces
    P  = firedrake.FunctionSpace(mesh, "CG", p) # to describe the solutions
    P0 = firedrake.FunctionSpace(mesh, "DG", 0) # to compute element area/volume 
    P_ten  = firedrake.TensorFunctionSpace(mesh, "CG", p) # to describe the solutions
    P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1) # to describe the mesh space
    V = P*P_ten # for solutions: phi and sigma=H(phi) (H is the Hessian)

    # Create variables used during the mesh movement
    theta = firedrake.Constant(0.0)
    monitor = firedrake.Function(P, name="Monitor function") 
    monitor.interpolate(monitor_function(mesh)) # initialize the monitor function
    volume = firedrake.Function(P0, name="Mesh volume")
    volume.interpolate(ufl.CellVolume(mesh))
    original_volume = firedrake.Function(volume) # initial cell volumes
    total_volume = firedrake.assemble(firedrake.Constant(1.0)*dx) # initial total volume
    L_P0 = firedrake.TestFunction(P0)*monitor*dx # used to compute the volume
    grad_phi_proj = firedrake.Function(P1_vec) # grad phi (on the mesh space) obtained by L2 projection)
    grad_phi_mesh = firedrake.Function(mesh.coordinates) # grad phi on the mesh space

    # Create functions to the weak formulation
    phisigma = firedrake.Function(V)
    phi, sigma = phisigma.split()
    phisigma_old = firedrake.Function(V)
    phi_old, sigma_old = phisigma_old.split()
    psi, tau = firedrake.TestFunctions(V)

    # Define n and I
    n = ufl.FacetNormal(mesh)
    I = ufl.Identity(dim)

    # Setup residuals used in the diagnostic FIXME it could be simplified
    theta_form = monitor*ufl.det(I + sigma_old)*dx
    residual = monitor*ufl.det(I + sigma_old) - theta
    residual_l2_form = psi*residual*dx
    norm_l2_form = psi*theta*dx

    # Set the equidistributor 
    phi, sigma = firedrake.split(phisigma) # FIXME testing
    F = ufl.inner(tau, sigma)*dx \
        + ufl.dot(ufl.div(tau), ufl.grad(phi))*dx \
        - (tau[0, 1]*n[1]*phi.dx(0) + tau[1, 0]*n[0]*phi.dx(1))*ds \
        - psi*(monitor*ufl.det(I + sigma) - theta)*dx
  
    # Create L2 projector to update the mesh coordinates
    def l2_projector(): # {{{
        # Using global variables because this function is called inside NonlinearSolver
        
        u_cts = firedrake.TrialFunction(P1_vec)
        v_cts = firedrake.TestFunction(P1_vec)

        # Domain interior
        a = ufl.inner(v_cts, u_cts)*dx
        L = ufl.inner(v_cts, ufl.grad(phi_old))*dx

        # Enforce no movement normal to boundary
        n = ufl.FacetNormal(mesh)
        bcs = []
        for i in mesh.exterior_facets.unique_markers:
            if fix_boundary_nodes:
                bcs.append(firedrake.DirichletBC(P1_vec, 0, i))
                continue

            # Check for axis-aligned boundaries
            _n = [firedrake.assemble(abs(n[j])*ds(i)) for j in range(dim)]
            if np.allclose(_n, 0.0):
                raise ValueError(f"Invalid normal vector {_n}")
            else:
                if dim != 2:
                    raise NotImplementedError  # TODO
                if np.isclose(_n[0], 0.0):
                    bcs.append(firedrake.DirichletBC(P1_vec.sub(1), 0, i))
                    continue
                elif np.isclose(_n[1], 0.0):
                    bcs.append(firedrake.DirichletBC(P1_vec.sub(0), 0, i))
                    continue
            
            # Enforce no mesh movement normal to boundaries
            a_bc = ufl.dot(v_cts, n)*ufl.dot(u_cts, n)*ds
            L_bc = ufl.dot(v_cts, n)*firedrake.Constant(0.0)*ds
            bcs.append(firedrake.EquationBC(a_bc == L_bc, grad_phi_proj, i))

            # Allow tangential movement, but only up until the end of boundary segments
            s = ufl.perp(n)
            a_bc = ufl.dot(v_cts, s)*ufl.dot(u_cts, s)*ds
            L_bc = ufl.dot(v_cts, s)*ufl.dot(ufl.grad(phi_old), s)*ds
            edges = set(mesh.exterior_facets.unique_markers)
            if len(edges) == 0:
                bbc = None  # Periodic case
            else:
                from warnings import warn
                warn('Have you checked that all straight line segments are uniquely tagged?')
                corners = [(i, j) for i in edges for j in edges.difference([i])]
                bbc = firedrake.DirichletBC(P1_vec, 0, corners)
            bcs.append(firedrake.EquationBC(a_bc == L_bc, grad_phi_proj, i, bcs=bbc))
            
            # TODO force regions of receivers/sources to be fixed

        # Create solver
        problem = firedrake.LinearVariationalProblem(a, L, grad_phi_proj, bcs=bcs)

        return firedrake.LinearVariationalSolver(problem, solver_parameters=_cg)
    #}}}
    # Update x
    def update_x(): # {{{
        """
        Update the coordinate :class:`Function` using
        the recovered gradienti (grad_phi_proj).
        """
        try:
            grad_phi_mesh.assign(grad_phi_proj)
        except Exception:
            grad_phi_mesh.interpolate(grad_phi_proj)
        x.assign(xi + grad_phi_mesh)  # x = ξ + grad(φ)
        return x
    #}}}
    # Update monitor function
    l2_projector_solver = l2_projector()
    def update_monitor(cursol): # {{{
        """
        Callback for updating the monitor function.
        """
        # using global variables because this is called inside the NonlinearSolver
        with phisigma_old.dat.vec as v:
            cursol.copy(v)
        l2_projector_solver.solve() # update grad_phi_proj 
        mesh.coordinates.assign(update_x()) # update x = xi + grad_phi  # FIXME check if the spaces are changed here
        monitor.interpolate(monitor_function(mesh)) # update monitor function FIXME check if should be done in xi 
        mesh.coordinates.assign(xi) # FIXME check if the spaces are changed here
        theta.assign(firedrake.assemble(theta_form)*total_volume**(-1))
   #}}} 

    # Custom preconditioner
    phi, sigma = firedrake.TrialFunctions(V) # phi and sigma here are trial functions
    Jp = ufl.inner(tau, sigma)*dx \
        + phi*psi*dx \
        + ufl.inner(ufl.grad(phi), ufl.grad(psi))*dx

    # Setup the variational problem
    problem = firedrake.NonlinearVariationalProblem(F, phisigma, Jp=Jp)
    nullspace = firedrake.MixedVectorSpaceBasis(V, [firedrake.VectorSpaceBasis(constant=True), V.sub(1)])
    sp = _serial_qn if firedrake.COMM_WORLD.size == 1 else _parallel_qn
    sp['snes_atol'] = rtol
    sp['snes_max_it'] = maxiter
    equidistributor = firedrake.NonlinearVariationalSolver(problem,
                                                           nullspace=nullspace,
                                                           transpose_nullspace=nullspace,
                                                           pre_function_callback=update_monitor,
                                                           pre_jacobian_callback=update_monitor,
                                                           solver_parameters=sp)
    
    # Set data to output the diagnostics of the Quasi-Newton solver
    def diagnostics(): # {{{
        """
        Compute diagnostics:
          1) the ratio of the smallest and largest element volumes;
          2) equidistribution of elemental volumes;
          3) relative L2 norm residual.
        """
        # using global variables because this is called inside the NonlinearSolver
        v = volume.vector().gather()
        minmax = v.min()/v.max()
        mean = v.sum()/v.max()
        w = v.copy() - mean
        w *= w
        std = np.sqrt(w.sum()/w.size)
        equi = std/mean
        residual_l2 = firedrake.assemble(residual_l2_form).dat.norm
        norm_l2 = firedrake.assemble(norm_l2_form).dat.norm
        residual_l2_rel = residual_l2/norm_l2
        return minmax, residual_l2_rel, equi
    #}}}

    # Define a function to print the progress of the Quasi-Newton solver
    @no_annotations
    def print_solver_progress(snes, i, rnorm): # {{{
        cursol = snes.getSolution()
        update_monitor(cursol)
        mesh.coordinates.assign(update_x()) # FIXME check if it should be xi, and if this changes the space
        firedrake.assemble(L_P0, tensor=volume)
        volume.assign(volume/original_volume)
        mesh.coordinates.assign(xi) 
        minmax, residual, equi = diagnostics()
        PETSc.Sys.Print(f"{i:4d}"
                        f"   Min/Max {minmax:10.4e}"
                        f"   Residual {residual:10.4e}"
                        f"   Equidistribution {equi:10.4e}")
    #}}}

    snes = equidistributor.snes
    snes.setMonitor(print_solver_progress)

    # Solve the Monge-Ampere using Quasi-Newton method
    try:
        import time
        ti=time.time()
        equidistributor.solve()
        tf=time.time()
        print("time equidistributor solver="+str(tf-ti))
        i = snes.getIterationNumber()
        PETSc.Sys.Print(f"Converged in {i} iterations.")
    except firedrake.ConvergenceError:
        i = snes.getIterationNumber()
        raise firedrake.ConvergenceError(f"Failed to converge in {i} iterations.")
    
    mesh.coordinates.assign(update_x()) #FIXME maybe return only the coordinates x
    return i
