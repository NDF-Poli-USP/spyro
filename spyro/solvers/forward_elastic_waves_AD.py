import firedrake as fire
from ..domains import quadrature, space
from . import helpers
from ..utils import utils
from .. import tools

# Note this turns off non-fatal warnings
fire.set_log_level(fire.ERROR)


def forward_elastic_waves_AD(
    model,
    mesh,
    comm,
    rho,
    lamb,
    mu,
    excitations,
    wavelet,
    receivers,
    source_num=0,
    output=False, 
    **kwargs
):
    """Secord-order in time fully-explicit scheme
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
        The rho value (density) interpolated onto the mesh (in g/cc == Gt/km3)
    lamb: Firedrake.Function
        The lambda value (1st Lame parameter) interpolated onto the mesh (in GPa)
      mu: Firedrake.Function
        The mu value (2nd Lame parameter) interpolated onto the mesh (in GPa)
    excitations: A list Firedrake.Functions
    wavelet: array-like
        Time series data that's injected at the source location.
    receivers: A :class:`spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    source_num: `int`, optional
        The source number you wish to simulate
    output: `boolean`, optional
        Whether or not to write results to pvd files.

    Returns
    -------
    usol: list of Firedrake.Functions
        The full field solution (displacements) at `fspool` timesteps
    uzsol_recv: array-like
    uxsol_recv: array-like
    uysol_recv: array-like
        The solution (displacement at each direction) interpolated to the receivers at all timesteps

    """
 
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    excitations.current_source = source_num
    nt = int(tf / dt)  # number of timesteps
    element = space.FE_method(mesh, method, degree)
    V = fire.VectorFunctionSpace(mesh, element)
    qr_x, qr_s, _ = quadrature.quadrature_rules(V)
    
    params = parameters(method, mesh)
    rec_loc = model["acquisition"]["receiver_locations"]
    h_min = tools.min_equilateral_distance(mesh,  fire.FunctionSpace(mesh, element), rec_loc)
       
    # if requested, set the output file
    if output:
        outfile = helpers.create_output_file("forward_elastic_waves_AD.pvd", comm, source_num)

    # create the trial and test functions
    # for typical CG/KMV FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)  # Test Function
    f = fire.Function(V, name="f")
    X = fire.Function(V, name="X")
    u_n = fire.Function(V, name="u_n")      # n
    u_nm1 = fire.Function(V, name="u_nm1")  # n-1

    # strain tensor
    def D(w):
        return 0.5 * (fire.grad(w) + fire.grad(w).T)

    du2_dt2 = ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt ** 2))
    # mass matrix
    m = rho * fire.inner(du2_dt2, v) * fire.dx(rule=qr_x)  # explicit
    # stiffness matrix
    a = lamb * fire.tr(D(u_n)) * fire.tr(D(v)) * fire.dx(rule=qr_x) 
    b = 2.0 * mu * fire.inner(D(u_n), D(v)) * fire.dx(rule=qr_x)
    # external forcing form 
    f_term = fire.inner(f, v) * fire.dx(rule=qr_x)
   
    # absorbing boundary conditions {{{
    nf = 0  # it enters as a Neumann-type BC
    if model["BCs"]["status"]:  # to turn on any type of BC
        boundary_condition(model, mesh, lamb, mu, rho, dt, u_n, 
                           u_nm1, v, output, element, qr_s, qr_x, nf)
    # weak formulation written as F=0
    F = m + a + b - f_term + nf

    # retrieve the lhs and rhs terms from F
    lhs_ = fire.lhs(F)
    rhs_ = fire.rhs(F)

    # create functions such that we solve for X in A X = B
    X = fire.Function(V)
    problem = fire.LinearVariationalProblem(lhs_, rhs_, X)
    solver = fire.LinearVariationalSolver(problem, solver_parameters=params)            

    # define the output solution over the entire domain (usol) and at the receivers (usol_recv)
    t = 0.0
    save_step = 0
    usol = [
            fire.Function(V, name="Displacement") 
            for t in range(nt) if t % fspool == 0
            ] # vectorized, includes uz, ux, and uy
    
    uzsol_recv = []  # u along the z direction
    uxsol_recv = []  # u along the x direction
    uysol_recv = []  # u along the y direction

    J0 = 0.0
    vertex_only_mesh = receivers.vertex_only_mesh_interpolator
    interpolator, P = vertex_only_mesh(u_nm1, elastic=True)
    # run forward in time
    for step in range(nt):

        solver.solve()
        f = excitations.apply_source(f, wavelet[step]/(h_min*h_min), elastic=True)
        # .apply_radial_source(f, wavelet[step]/(h_min*h_min))

        # deal with absorbing boundary layers
        if model["BCs"]["status"] and model["BCs"]["abl_bc"] == "gaussian-taper":
            # FIXME check if all these terms are needed and if dim==3
            X.sub(0).assign(X.sub(0)*gp) 
            X.sub(1).assign(X.sub(1)*gp)
            u_nm1.sub(0).assign(u_nm1.sub(0)*gp)
            u_nm1.sub(1).assign(u_nm1.sub(1)*gp)
            u_n.sub(0).assign(u_n.sub(0)*gp)
            u_n.sub(1).assign(u_n.sub(1)*gp)
        
        fwi = kwargs.get("fwi")
        # interpolate the solution at the receiver points
        rec = fire.Function(P)
        interpolator.interpolate(output=rec)
        
        if fwi:
            p_true_rec = kwargs.get("true_rec")
            # print(rec.sub(0).dat.data[:])
            J0 += objective_func(
                rec,
                p_true_rec,
                step,
                dt,
                P)

        uzsol_recv.append(rec.sub(0).dat.data) 
        uxsol_recv.append(rec.sub(1).dat.data) 
        # save the solution if requested for this time step
        if step % fspool == 0:
            usol[save_step].assign(X)
            save_step += 1

        if step % nspool == 0:
            assert (
                fire.norm(u_n) < 1 
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                u_n.rename("Displacement")
                outfile.write(u_n, time=t)
            #if t > 0: 
                #helpers.display_progress(comm, t) #  FIXME uncomment it

        # update u^(n-1) and u^(n)
        u_nm1.assign(u_n)
        u_n.assign(X)

        t = step * float(dt)

    if fwi:
        return usol, uzsol_recv, uxsol_recv, uysol_recv, J0
    else:
        return usol, uzsol_recv, uxsol_recv, uysol_recv


def objective_func(p_rec, p_true_rec, IT, dt, P):
    true_rec = fire.Function(P)
    true_rec.sub(0).dat.data[:] = p_true_rec[0][IT]
    true_rec.sub(1).dat.data[:] = p_true_rec[1][IT]
    J = 0.5 * fire.assemble(fire.inner(true_rec-p_rec, true_rec-p_rec) * fire.dx)
    return J


def boundary_condition(model, mesh, lamb, mu, rho, dt, u_n, u_nm1, v, output, element, qr_s, qr_x, nf):
    dim = model["opts"]["dimension"]
    if dim == 2:
        z, x = fire.SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = fire.SpatialCoordinate(mesh)

    bc_defined = False
    x1 = 0.0                   # z-x origin
    x2 = model["mesh"]["Lx"]   # effective width of the domain, excluding the absorbing layers
    z2 = -model["mesh"]["Lz"]  # effective depth of the domain, excluding the absorbing layers
    lx = model["BCs"]["lx"]    # width of the absorbing layer
    lz = model["BCs"]["lz"]    # depth of the absorbing layer

    # damping at outer boundaries (-x,+x,-z,+z)
    if model["BCs"]["outer_bc"] == "non-reflective" and model["BCs"]["abl_bc"] != "alid":
        # get normal and tangent vectors
        n = fire.FacetNormal(mesh)
        t = fire.perp(n)

        c_p = ((lamb + 2.*mu)/rho)**0.5
        c_s = (mu/rho)**0.5
        C = c_p * fire.outer(n, n) + c_s * fire.outer(t, t)
        
        nf += rho * fire.inner(C * ((u_n - u_nm1) / dt), v) * fire.ds(rule=qr_s)  # backward-difference scheme 
        bc_defined = True
    else:
        if model["BCs"]["outer_bc"] == "non-reflective" and model["BCs"]["abl_bc"] == "alid":
            print("WARNING: [BCs][outer_bc] = non-reflectie AND [BCs][abl_bc] = alid. Ignoring [BCs][outer_bc].")

    # absorbing layer with increasing damping (ALID) (-x,+x,-z)
    if model["BCs"]["abl_bc"] == "alid":
        cmax = 10*((lamb + 2*mu)/rho)**0.5 # FIXME testing c_p
        #cmax = (mu/rho)**0.5 # FIXME testing c_s
        p = 3  # FIXME define p
        g = fire.conditional(x < x1, (abs(x1-x)/lx)**p, 0.0)  # assuming that all x<x1 belongs to abs layer
        g = fire.conditional(x > x2, (abs(x2-x)/lx)**p, g)    # assuming that all x>x2 belongs to abs layer
        g = fire.conditional(z < z2, (abs(z2-z)/lz)**p, g)    # assuming that all z<z2 belongs to abs layer
        g = fire.conditional(fire.And(x < x1, z < z2), ((((x1-x)**2.0+(z2-z)**2.0)**0.5)/min(lx,lz) )**p, g)
        g = fire.conditional(fire.And(x > x2, z < z2), ((((x2-x)**2.0+(z2-z)**2.0)**0.5)/min(lx,lz) )**p, g)
        G = fire.FunctionSpace(mesh, element)
        alid_mask = fire.Function(G, name="Damping_coefficient").interpolate(g)
        if output:
            fire.File("damping_coefficient_alid.pvd").write(alid_mask)
        nf += alid_mask * cmax * fire.inner( ((u_n - u_nm1) / dt), v) * fire.dx(rule=qr_x) #FIXME check if it should be unp1-un
        #nf = alid_mask * cmax * fire.inner( ((u - u_n) / dt) , v ) * fire.dx(rule=qr_x) #FIXME it produces the same result 
        bc_defined = True
    # absorbing layer with Gaussian taper 
    elif model["BCs"]["abl_bc"] == "gaussian-taper": 
        gamma = 2.               # FIXME define gamma
        g = fire.conditional(x < x1, fire.exp(-((x1-x)/gamma)**2.0), 1.0) # assuming that all x<x1 belongs to abs layer
        g = fire.conditional(x > x2, fire.exp(-((x2-x)/gamma)**2.0), g)   # assuming that all x>x2 belongs to abs layer
        g = fire.conditional(z < z2, fire.exp(-((z2-z)/gamma)**2.0), g)   # assuming that all z<z2 belongs to abs layer
        g = fire.conditional(fire.And(x < x1, z < z2), fire.exp(-((x1-x)/gamma)**2.0 -((z2-z)/gamma)**2.0), g)
        g = fire.conditional(fire.And(x > x2, z < z2), fire.exp(-((x2-x)/gamma)**2.0 -((z2-z)/gamma)**2.0), g)
        G = fire.FunctionSpace(mesh, element)
        gp = fire.Function(G, name="Gaussian_taper").interpolate(g)
        if output:
            fire.File("gaussian_taper.pvd").write(gp)
        bc_defined = True
    else:
        print("WARNING: absorbing boundary layer not defined ([BCs][abl_bc] = none).")
    
    if bc_defined == False:
        print("WARNING: [BCs][status] = True, but no boundary condition defined. Check your [BCs]")


def parameters(method, mesh):
    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif (
        method == "CG"
        and mesh.ufl_cell() != fire.quadrilateral
        and mesh.ufl_cell() != fire.hexahedron
    ):
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
        #params = {"ksp_type": "preonly", "pc_type": "lu"} # for direct solver
    elif method == "CG" and (
        mesh.ufl_cell() == fire.quadrilateral or mesh.ufl_cell() == fire.hexahedron
    ):
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")
    
    return params
