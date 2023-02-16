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

# @ensemble_forward_elastic_waves
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
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    excitations.current_source = source_num
    #delay  = model["acquisition"]["delay"]
    #dstep  = int(delay / dt)  # number of timesteps with source
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
        outfile = helpers.create_output_file("forward_elastic_waves_AD.pvd", comm, source_num)

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
   
    # absorbing boundary conditions {{{
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
            n = FacetNormal(mesh)
            t = perp(n)

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
    #}}}

    # weak formulation written as F=0
    F = m + a - l + nf

    # retrieve the lhs and rhs terms from F
    lhs_ = lhs(F)
    rhs_ = rhs(F)

    # create functions such that we solve for X in A X = B
    X = Function(V)
    problem = LinearVariationalProblem(lhs_, rhs_, X)
    solver  = LinearVariationalSolver(problem, solver_parameters=params)            

    # define the output solution over the entire domain (usol) and at the receivers (usol_recv)
    t = 0.0
    save_step = 0
    usol = [Function(V, name="Displacement") for t in range(nt) if t % fspool == 0] # vectorized, includes uz, ux, and uy
    uzsol_recv = [] # u along the z direction
    uxsol_recv = [] # u along the x direction
    uysol_recv = [] # u along the y direction

    J0            = 0.0
    P             = VectorFunctionSpace(receivers, "DG", 0)
    interpolator  = Interpolator(u_np1, P)

    # run forward in time
    for step in range(nt):
        
        solver.solve() # keeping here to generate the same result generated by the forward_elastic_waves.py
 
        f = excitations.apply_radial_source(f, wavelet[step]/(h_min*h_min))

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
        
        fwi = kwargs.get("fwi")
        # interpolate the solution at the receiver points
        rec = Function(P)
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
            usol[save_step].assign(u_np1)
            save_step += 1

        if step % nspool == 0:
            assert (
                norm(u_n) < 1 
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                u_n.rename("Displacement")
                outfile.write(u_n, time=t)
            #if t > 0: 
                #helpers.display_progress(comm, t) #  FIXME uncomment it

        # update u^(n-1) and u^(n)
        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    if fwi:
        return usol, uzsol_recv, uxsol_recv, uysol_recv, J0
    else:
        return usol, uzsol_recv, uxsol_recv, uysol_recv


def objective_func(p_rec, p_true_rec, IT, dt, P):
    true_rec = Function(P)
    true_rec.sub(0).dat.data[:] = p_true_rec[0][IT]
    true_rec.sub(1).dat.data[:] = p_true_rec[1][IT]
    J = 0.5 * assemble(inner(true_rec-p_rec, true_rec-p_rec) * dx)
    return J
