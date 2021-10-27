from firedrake import *
from firedrake.assemble import create_assembly_callable

from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_gradient_elastic_waves 
from . import helpers
import sys

# Note this turns off non-fatal warnings
set_log_level(ERROR)

__all__ = ["gradient"] #FIXME check this


@ensemble_gradient_elastic_waves
def gradient_elastic_waves(
    model, mesh, comm, rho, lamb, mu, receivers, guess, residual_z, residual_x, residual_y, output=False, save_adjoint=False
):
    """Discrete adjoint with secord-order in time fully-explicit timestepping scheme
    with implementation of simple absorbing boundary conditions using 
    CG FEM with or without higher order mass lumping (KMV type elements).

    Parameters
    ----------
    model: Python `dictionary`
        Contains model options and parameters
    mesh: Firedrake.mesh object
        The 2D/3D triangular mesh
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    rho: Firedrake.Function
        The rho value (density) interpolated onto the mesh
    lamb: Firedrake.Function
        The lambda value (1st Lame parameter) interpolated onto the mesh
      mu: Firedrake.Function
        The mu value (2nd Lame parameter) interpolated onto the mesh
    
    receivers: A :class:`spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    guess: A list of Firedrake functions
        Contains the forward wavefield at a set of timesteps
    residual_z: array-like [timesteps][receivers]
    residual_x: array-like [timesteps][receivers]
    residual_y: array-like [timesteps][receivers]
        The difference between the observed and modeled data at
        the receivers, for each direction
    output: boolean
        optional, write the adjoint to disk (only for debugging)
    save_adjoint: A list of Firedrake functions
        Contains the adjoint at all timesteps

    Returns
    -------
    dJdl_local: A Firedrake.Function containing the gradient of
                the functional w.r.t. `lambda`
    dJdm_local: A Firedrake.Function containing the gradient of
                the functional w.r.t. `mu`
    adjoint: Optional, a list of Firedrake functions containing the adjoint

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    
    nt = int(tf / dt)  # number of timesteps

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif method == "CG":
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")

    #--------- start defintion of the adjoint problem ---------
    element = space.FE_method(mesh, method, degree)
    
    V = VectorFunctionSpace(mesh, element) # for adjoint approximation

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    receiver_locations = model["acquisition"]["receiver_locations"]

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)
    
    # if requested, set the output file
    if output:
        outfile = helpers.create_output_file("adjoint_elastic_waves.pvd", comm, 0)

    # create the trial and test functions for typical CG/KMV FEM in 2d/3d
    u = TrialFunction(V) # adjoint problem
    v = TestFunction(V)  # adjoint problem
    # create the external forcing function (vector-valued function)
    f = Function(V)

    # values of u (adjoint, vector-valued function) at different timesteps
    u_nm1 = Function(V) # timestep n-1
    u_n = Function(V, name="Adjoint")   # timestep n
    u_np1 = Function(V) # timestep n+1

    # strain tensor
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)

    # mass matrix (adjoint problem)
    m = (rho * inner((u - 2.0 * u_n + u_nm1),v) / Constant(dt ** 2)) * dx(rule=qr_x) # explicit
    # stiffness matrix (adjoint problem)
    a = lamb * tr(D(u_n)) * tr(D(v)) * dx(rule=qr_x) + 2.0 * mu * inner(D(u_n), D(v)) * dx(rule=qr_x)
    # external forcing form (adjoint problem)
    l = inner(f,v) * dx(rule=qr_x)

    # absorbing boundary conditions (adjoint problem)
    #FIXME check the Neumann BC because the adjoint formulation requires Null Neumann BC
    nf = 0 # it enters as a Neumann-type BC
    if model["BCs"]["status"]: # to turn on any type of BC
        bc_defined = False
        x1 = 0.0                 # z-x origin
        z1 = 0.0                 # z-x origin
        x2 = model["mesh"]["Lx"] # effective width of the domain, excluding the absorbing layers
        z2 =-model["mesh"]["Lz"] # effective depth of the domain, excluding the absorbing layers
        lx = model["BCs"]["lx"]  # width of the absorbing layer
        lz = model["BCs"]["lz"]  # depth of the absorbing layer
        
        # damping at outer boundaries (-x,+x,-z,+z)
        if model["BCs"]["outer_bc"] == "non-reflective" and model["BCs"]["abl_bc"] != "alid":
            # to get the normal vector
            #n = firedrake.FacetNormal(mesh)
            #print(assemble(inner(v, n) * ds))

            # FIXME keeping c_p for now, but it should be changed to a matrix form
            c_p = ((lamb + 2.*mu)/rho)**0.5
            #c_s = (mu/rho)**0.5
            nf = rho * c_p * inner( ((u_n - u_nm1) / dt) , v ) * ds(rule=qr_s) # backward-difference scheme 
            bc_defined = True
        else:
            if model["BCs"]["outer_bc"] == "non-reflective" and model["BCs"]["abl_bc"] == "alid":
                print("WARNING: [BCs][outer_bc] = non-reflectie AND [BCs][abl_bc] = alid. Ignoring [BCs][outer_bc].")    

        # absorbing layer with increasing damping (ALID) (-x,+x,-z)
        if model["BCs"]["abl_bc"] == "alid":
            cmax = 10*((lamb + 2*mu)/rho)**0.5 # FIXME testing c_p
            #cmax = (mu/rho)**0.5 # FIXME testing c_s
            p = 3 # FIXME define p
            g = conditional(x < x1, (abs(x1-x)/lx)**p, 0.0) # assuming that all x<x1 belongs to abs layer
            g = conditional(x > x2, (abs(x2-x)/lx)**p, g)   # assuming that all x>x2 belongs to abs layer
            g = conditional(z < z2, (abs(z2-z)/lz)**p, g)   # assuming that all z<z2 belongs to abs layer
            g = conditional(And(x < x1, z < z2), ( (((x1-x)**2.0+(z2-z)**2.0)**0.5)/min(lx,lz) )**p, g)
            g = conditional(And(x > x2, z < z2), ( (((x2-x)**2.0+(z2-z)**2.0)**0.5)/min(lx,lz) )**p, g)
            G = FunctionSpace(mesh, element)
            alid_mask = Function(G, name="Damping_coefficient").interpolate(g)
            if output:
                File("damping_coefficient_alid.pvd").write(alid_mask)
            nf = alid_mask * cmax * inner( ((u_n - u_nm1) / dt) , v ) * dx(rule=qr_x) #FIXME check if it should be unp1-un
            #nf = alid_mask * cmax * inner( ((u - u_n) / dt) , v ) * dx(rule=qr_x) #FIXME it produces the same result 
            bc_defined = True
        # absorbing layer with Gaussian taper 
        elif model["BCs"]["abl_bc"] == "gaussian-taper":
            gamma = 2.               # FIXME define gamma
            g = conditional(x < x1, exp(-((x1-x)/gamma)**2.0), 1.0) # assuming that all x<x1 belongs to abs layer
            g = conditional(x > x2, exp(-((x2-x)/gamma)**2.0), g)   # assuming that all x>x2 belongs to abs layer
            g = conditional(z < z2, exp(-((z2-z)/gamma)**2.0), g)   # assuming that all z<z2 belongs to abs layer
            g = conditional(And(x < x1, z < z2), exp(-((x1-x)/gamma)**2.0 -((z2-z)/gamma)**2.0), g)
            g = conditional(And(x > x2, z < z2), exp(-((x2-x)/gamma)**2.0 -((z2-z)/gamma)**2.0), g)
            G = FunctionSpace(mesh, element)
            gp = Function(G, name="Gaussian_taper").interpolate(g)
            if output:
                File("gaussian_taper.pvd").write(gp)
            bc_defined = True
        else:
            print("WARNING: absorbing boundary layer not defined ([BCs][abl_bc] = none).")

        if bc_defined == False:
            print("WARNING: [BCs][status] = True, but no boundary condition defined. Check your [BCs]")

    # weak formulation written as F=0 (adjoint problem)
    F = m + a - l + nf

    # retrieve the lhs and rhs terms from F
    lhs_ = lhs(F)
    rhs_ = rhs(F)

    # FIXME DirichletBC does not help prevent oscilations when mu=0
    #bc = DirichletBC(V.sub(0), 0., (1,2,3,4) )
    bc = DirichletBC(V, (0.,0.), (1,2,3,4) )

    # create functions such that we solve for X in A X = B (adjoint problem)
    X = Function(V)
    B = Function(V)
    A = assemble(lhs_, mat_type="matfree")
    #A = assemble(lhs_) # for direct solver

    # set the linear solver for A
    solver = LinearSolver(A, solver_parameters=params)
    
    if save_adjoint:
        adjoint = [Function(V, name="adjoint_elastic_waves") for t in range(nt)]

    #--------- end defintion of the adjoint problem ---------

    #--------- start defintion of the gradient problem ---------
    H = FunctionSpace(mesh, element) # scalar space for gradient approximation
    
    dJdl = Function(H, name="gradient_lambda")
    dJdm = Function(H, name="gradient_mu")
    
    ug = TrialFunction(H)
    vg = TestFunction(H)
    
    ufor = Function(V)  # forward, auxiliarly function for the gradient computation by L2 inner product 
    uadj = Function(V)  # adjoint, auxiliarly function for the gradient computation by L2 inner product
    
    # mass matrix (gradient computation by L2 inner product)
    mg = ug * vg * dx(rule=qr_x)
    # gradient 
    agl = tr(D(ufor)) * tr(D(uadj)) * vg * dx(rule=qr_x) # w.r.t. lambda
    agm = 2.0 * inner(D(ufor), D(uadj)) * vg * dx(rule=qr_x) # w.r.t. mu

    # weak formulationd written as F=0 (gradient problem)
    Fgl = mg - agl  # for dJdl (gradient w.r.t. lambda)
    Fgm = mg - agm  # for dJdm (gradient w.r.t. mu)
   
    # retrieve the lhs and rhs terms from F
    lhsFgl, rhsFgl = lhs(Fgl), rhs(Fgl)
    lhsFgm, rhsFgm = lhs(Fgm), rhs(Fgm)

    dJdl_inc = Function(H) # increment of dJdl
    dJdm_inc = Function(H) # increment of dJdm
    
    dJdl_prob = LinearVariationalProblem(lhsFgl, rhsFgl, dJdl_inc) 
    dJdm_prob = LinearVariationalProblem(lhsFgm, rhsFgm, dJdm_inc)
    
    if method == "KMV":
        # to solve for dJdl
        dJdl_solver = LinearVariationalSolver(
            dJdl_prob,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                "mat_type": "matfree",
            },
        )
        # to solve for dJdm
        dJdm_solver = LinearVariationalSolver(
            dJdm_prob,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                "mat_type": "matfree",
            },
        )
    elif method == "CG":
        # to solve for dJdl
        dJdl_solver = LinearVariationalSolver(
            dJdl_prob,
            solver_parameters={
                "mat_type": "matfree",
            },
        )
        # to solve for dJdm
        dJdm_solver = LinearVariationalSolver(
            dJdm_prob,
            solver_parameters={
                "mat_type": "matfree",
            },
        )
    #--------- end defintion of the gradient problem ---------

    # run backward in time
    for step in range(nt - 1, -1, -1):
        t = step * float(dt)
        
        # assemble the rhs term to update the forcing FIXME assemble here or after apply source?
        B = assemble(rhs_, tensor=B)
        bc.apply(B) #FIXME for Dirichlet BC

        # apply the residual evaluated at the receivers as external forcing (sources)
        #FIXME check the sign of f / residuals
        f = receivers.apply_receivers_as_radial_source(f, residual_z, residual_x, residual_y, step)
        #File("f.pvd").write(f)
        #sys.exit("exiting")

        # solve and assign X onto solution u 
        solver.solve(X, B)
        u_np1.assign(X)

        # gradient computation: only compute for snaps that were saved
        if step % fspool == 0:
            # compute the gradient increment
            uadj.assign(u_np1)
            ufor.assign(guess.pop())
            # solve the L2 inner product 
            dJdl_solver.solve()
            dJdm_solver.solve()
            # add to the gradient
            dJdl += dJdl_inc
            dJdm += dJdm_inc

        # update u^(n-1) and u^(n) FIXME check if the assign is here or after output write (also check the forward prob.)
        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        if step % nspool == 0:
            if output:
                outfile.write(u_n, time=t)
            if save_adjoint: 
                adjoint.append(u_n) 
            helpers.display_progress(comm, t)

    if save_adjoint:
        return dJdl, dJdm, adjoint
    else:
        return dJdl, dJdm
