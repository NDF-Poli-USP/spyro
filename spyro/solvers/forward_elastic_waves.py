from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from . import helpers

# Note this turns off non-fatal warnings
set_log_level(ERROR)

@ensemble_forward
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
    lamb: Firedrake.Function
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
        The full field solution at `fspool` timesteps
    usol_recv: array-like
        The solution interpolated to the receivers at all timesteps

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    #PML = model["BCs"]["status"] FIXME
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
    a = lamb * tr(D(u_n)) * tr(D(v)) * dx + 2.0 * mu * inner(D(u_n),D(v)) * dx(rule=qr_x)
    # external forcing form 
    l = inner(f,v) * dx(rule=qr_x) 
    
    #nf = 0 FIXME BC
    #if model["BCs"]["outer_bc"] == "non-reflective":
    #    nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)
    
    # the weak formulation written as F=0
    F = m + a - l 

    # retrieve the lhs and rhs terms from F
    lhs_ = lhs(F)
    rhs_ = rhs(F)

    # create functions such that we solve for X in A X = B
    X = Function(V)
    B = Function(V)
    A = assemble(lhs_, mat_type="matfree")
    
    # set the linear solver for A
    solver = LinearSolver(A, solver_parameters=params)

    # define the output solution over the entire domain (usol) and at the receivers (usol_recv)
    t = 0.0
    save_step = 0
    usol = [Function(V, name="Displacement") for t in range(nt) if t % fspool == 0]
    usol_recv = []

    # run forward in time
    for step in range(nt):
        # assemble the rhs term to update the forcing FIXME assemble here or after apply source?
        B = assemble(rhs_, tensor=B)
        
        # apply source only in the x-direction for now 
        excitations.apply_source(f.sub(0), -1.*wavelet[step]) # x FIXME check the sign (-1)
        #excitations.apply_source(f.sub(1), wavelet[step]) # y FIXME check this in the future
        
        # solve and assign X onto solution u 
        solver.solve(X, B)
        u_np1.assign(X)

        # interpolate the solution at the receiver points
        usol_recv.append(receivers.interpolate(u_np1.dat.data_ro_with_halos[:])) # FIXME check this

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
            if t > 0:
                helpers.display_progress(comm, t)

        # update u^(n-1) and u^(n)
        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    # prepare to return
    usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
    usol_recv = utils.communicate(usol_recv, comm)

    return usol, usol_recv
