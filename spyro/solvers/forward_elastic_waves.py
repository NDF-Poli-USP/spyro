from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward_elastic_waves
from . import helpers

import sys
import time

# Note this turns off non-fatal warnings
set_log_level(ERROR)

@ensemble_forward_elastic_waves
def forward_elastic_waves(
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
        The rho value (density) interpolated onto the mesh
    lamb: Firedrake.Function
        The lambda value (1st Lame parameter) interpolated onto the mesh
      mu: Firedrake.Function
        The mu value (2nd Lame parameter) interpolated onto the mesh
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
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    excitations.current_source = source_num

    nt = int(tf / dt)  # number of timesteps

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif (
        method == "CG"
        and mesh.ufl_cell() != quadrilateral
        and mesh.ufl_cell() != hexahedron
    ):
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
        #params = {"ksp_type": "preonly", "pc_type": "lu"} # for direct solver
    elif method == "CG" and (
        mesh.ufl_cell() == quadrilateral or mesh.ufl_cell() == hexahedron
    ):
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")

    element = space.FE_method(mesh, method, degree)

    V = VectorFunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)
    
    # if requested, set the output file
    if output:
        outfile = helpers.create_output_file("forward_elastic_waves.pvd", comm, source_num)

    # create the trial and test functions for typical CG/KMV FEM in 2d/3d
    u = TrialFunction(V)
    v = TestFunction(V)
    # create the external forcing function (vector-valued function)
    f = Function(V)
    
    # values of u at different timesteps
    u_nm1 = Function(V) # timestep n-1
    u_n = Function(V)   # timestep n
    u_np1 = Function(V) # timestep n+1

    # strain tensor
    def D(w):   
        return 0.5 * (grad(w) + grad(w).T)

    # mass matrix 
    m = (rho * inner((u - 2.0 * u_n + u_nm1),v) / Constant(dt ** 2)) * dx(rule=qr_x) # explicit
    # stiffness matrix
    a = lamb * tr(D(u_n)) * tr(D(v)) * dx(rule=qr_x) + 2.0 * mu * inner(D(u_n), D(v)) * dx(rule=qr_x)
    # external forcing form 
    l = inner(f,v) * dx(rule=qr_x) 
   
    # absorbing boundary conditions
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
            # get normal and tangent vectors 
            n = firedrake.FacetNormal(mesh)
            t = firedrake.perp(n)

            c_p = ((lamb + 2.*mu)/rho)**0.5
            c_s = (mu/rho)**0.5
            C = c_p * outer(n,n) + c_s * outer(t,t)
            
            nf = rho * inner( C * ((u_n - u_nm1) / dt), v ) * ds(rule=qr_s) # backward-difference scheme 
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

    #n = firedrake.FacetNormal(mesh)
    #t = firedrake.perp(n)
    #print(assemble(inner(v, n) * ds))
    #print(assemble(inner(v, t) * ds))
    #https://fenicsproject.discourse.group/t/integrate-over-edges/1140/8

    # weak formulation written as F=0
    F = m + a - l + nf 

    # retrieve the lhs and rhs terms from F
    lhs_ = lhs(F)
    rhs_ = rhs(F)
    
    # FIXME DirichletBC does not help prevent oscilations when mu=0
    #bc = DirichletBC(V.sub(0), 0., (1,2,3,4) )
    #bc = DirichletBC(V, (0.,0.), (1,2,3,4) )

    # create functions such that we solve for X in A X = B
    X = Function(V)
    B = Function(V)
    A = assemble(lhs_, mat_type="matfree")
    #A = assemble(lhs_) # for direct solver
    
    # set the linear solver for A
    solver = LinearSolver(A, solver_parameters=params)

    # define the output solution over the entire domain (usol) and at the receivers (usol_recv)
    t = 0.0
    save_step = 0
    usol = [Function(V, name="Displacement") for t in range(nt) if t % fspool == 0] # vectorized, includes uz, ux, and uy
    uzsol_recv = [] # u along the z direction
    uxsol_recv = [] # u along the x direction
    uysol_recv = [] # u along the y direction

    def delta_expr(xs, zs, x, z, sigma_x=500):
        sigma_x = Constant(sigma_x)
        return exp(-sigma_x * ((x - xs) ** 2 + (z - zs) ** 2))

    radial_source=0
    if radial_source==1: # FIXME keeping this old code here for now
        xs = model["acquisition"]["source_pos"][0][1]       
        zs = model["acquisition"]["source_pos"][0][0]       
        S = FunctionSpace(mesh, element)
        tol = 0.00001
        source_x = Function(S, name="source_x").interpolate(
                            delta_expr(xs, zs, x, z) * (x-xs)/(tol + ((x-xs)**2.+(z-zs)**2.)**0.5)
                            )
        source_z = Function(S, name="source_z").interpolate(
                            delta_expr(xs, zs, x, z) * (z-zs)/(tol + ((x-xs)**2.+(z-zs)**2.)**0.5)
                            )
        File("source_x.pvd", comm=comm.comm).write(source_x)
        File("source_z.pvd", comm=comm.comm).write(source_z)
        sys.exit("Exit without running")

    # FIXME testing
    #outfile2 = helpers.create_output_file("p-wave.pvd", comm, source_num)
    #outfile3 = helpers.create_output_file("s-wave.pvd", comm, source_num)
    #S = FunctionSpace(mesh, element)
    AD = True
    if AD:
        J0           = 0.0
        J1           = 0.0
        P            = FunctionSpace(receivers, "DG", 0)
        interpolator = Interpolator(u_np1.sub(0), P)

    # run forward in time
    for step in range(nt):
        # assemble the rhs term to update the forcing FIXME assemble here or after apply source?
        B = assemble(rhs_, tensor=B)
        #bc.apply(B) #FIXME for Dirichlet BC

        #start = time.time()
        if radial_source==1: # FIXME testing a radial source here for now
            f.sub(0).assign(wavelet[step]*source_z)
            f.sub(1).assign(wavelet[step]*source_x)
        else:
            #FIXME let the user decide which approach will be used 
            #excitations.apply_source(f.sub(0), -1.*wavelet[step]) # z 
            #excitations.apply_source(f.sub(1), wavelet[step]) # x 
            f = excitations.apply_radial_source(f, wavelet[step])
            #f.sub(1).assign(Function(S).interpolate(sin(x))) # only P-wave
        #end = time.time()
        #print(end - start)

        # solve and assign X onto solution u 
        solver.solve(X, B)
        u_np1.assign(X)

        # deal with absorbing boundary layers
        if model["BCs"]["status"] and model["BCs"]["abl_bc"] == "gaussian-taper":
            # FIXME check if all these terms are needed and if dim==3
            u_np1.sub(0).assign(u_np1.sub(0)*gp) 
            u_np1.sub(1).assign(u_np1.sub(1)*gp)
            u_nm1.sub(0).assign(u_nm1.sub(0)*gp)
            u_nm1.sub(1).assign(u_nm1.sub(1)*gp)
            u_n.sub(0).assign(u_n.sub(0)*gp)
            u_n.sub(1).assign(u_n.sub(1)*gp)

        # interpolate the solution at the receiver points
        if AD: # 2D por enquanto
            rec = Function(P)
            interpolator.interpolate(output=rec)
            
            print(rec.dat.data_ro)
            quit()
            uzsol_recv.append(rec.sub(0).dat.data) 
            uxsol_recv.append(rec.sub(1).dat.data)
           
            # fwi = kwargs.get("fwi")

            # if fwi:    
            #     true_rec = kwargs.get("true_rec")
            #     rec0             = Function(P)
            #     rec1             = Function(P)
            #     rec0.dat.data[:] = true_rec[0]
            #     rec1.dat.data[:] = true_rec[1]
            #     J0 += 0.5 * assemble(inner(rec0-rec.sub(0), rec0-rec.sub(0)) * dx)
            #     J1 += 0.5 * assemble(inner(rec1-rec.sub(0), rec0-rec.sub(0)) * dx)
        else:   
            uzsol_recv.append(receivers.interpolate(u_np1.sub(0).dat.data_ro_with_halos[:])) # z direction
            uxsol_recv.append(receivers.interpolate(u_np1.sub(1).dat.data_ro_with_halos[:])) # x direction
            if dim==3:
                uysol_recv.append(receivers.interpolate(u_np1.sub(2).dat.data_ro_with_halos[:])) # y direction

        # save the solution if requested for this time step
        if step % fspool == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if step % nspool == 0:
            assert (
                norm(u_n) < 1 # FIXME why u_n and not u_np1?
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                u_n.rename("Displacement")
                outfile.write(u_n, time=t)
                #FIXME testing
                #p_n = Function(S, name="p-wave").interpolate(tr(D(u_n)))
                #s_n = Function(S, name="s-wave").interpolate(curl(u_n)) #FIXME not sure this is right
                #outfile2.write(p_n, time=t)
                #outfile3.write(s_n, time=t)
            #if t > 0: 
                # helpers.display_progress(comm, t) FIXME uncomment it

        # update u^(n-1) and u^(n)
        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)
    
    # prepare to return
    # uzsol_recv = helpers.fill(uzsol_recv, receivers.is_local, nt, receivers.num_receivers)
    # uxsol_recv = helpers.fill(uxsol_recv, receivers.is_local, nt, receivers.num_receivers)
    # uzsol_recv = utils.communicate(uzsol_recv, comm)
    # uxsol_recv = utils.communicate(uxsol_recv, comm)
    if dim==3:
        uysol_recv = helpers.fill(uysol_recv, receivers.is_local, nt, receivers.num_receivers)
        uysol_recv = utils.communicate(uysol_recv, comm)
    fwi=False
    if fwi:
        J = [J0, J1]
        return usol, uzsol_recv, uxsol_recv, uysol_recv, J
    else:
        return usol, uzsol_recv, uxsol_recv, uysol_recv
