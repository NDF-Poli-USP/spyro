from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..sources import FullRickerWavelet
from ..sources import MMS_time, timedependentSource
from . import helpers

set_log_level(ERROR)


def SSPRKMOD(
    model,
    mesh,
    comm,
    c,
    excitations,
    receivers,
    source_num=0,
    freq_index=0,
    output=False,
    G=1.0 #Added G only for debugging, will remove later
):
    """Secord-order in time fully-explicit Leapfrog scheme
    with implementation of a Perfectly Matched Layer (PML) using
    CG FEM with or without higher order mass lumping (KMV type elements).

    Parameters
    ----------
    model: Python `dictionary`
        Contains model options and parameters
    mesh: Firedrake.mesh object
        The 2D/3D triangular mesh
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
       c: Firedrake.Function
        The velocity model interpolated onto the mesh.
    excitations: A list Firedrake.Functions
        Each function contains an interpolated space function
        emulated a Dirac delta at the location of source `source_num`
    receivers: A :class:`Spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    source_num: `int`, optional
        The source number you wish to simulate
    freq_index: `int`, optional
        The index in the list of low-pass cutoff values
    output: `boolean`, optional
        Whether or not to write results to pvd files.

    Returns
    -------
    usol: list of Firedrake.Functions
        The full field solution at `fspool` timesteps
    usol_recv: array-like
        The solution interpolated to the receivers at all timesteps

    """

    if "amplitude" in model["acquisition"]:
        amp = model["acquisition"]["amplitude"]
    else:
        amp = 1
    freq = model["acquisition"]["frequency"]
    if "inversion" in model:
        freq_bands = model["inversion"]["freq_bands"]
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    delay = model["acquisition"]["delay"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]

    nt = int(tf / dt)  # number of timesteps
    dstep = int(delay / dt)  # number of timesteps with source

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif method == "CG" and mesh.ufl_cell() != quadrilateral and mesh.ufl_cell() != hexahedron :
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    elif method == "CG" and (mesh.ufl_cell() == quadrilateral or mesh.ufl_cell() == hexahedron ):
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")

    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)

    # typical CG in N-d
    W =  V* V
    dwdt_trial, dpdt_trial = TrialFunctions(W)
    q1, q2 = TestFunctions(W)

    w_n, p_n= Function(W).split()



    is_local = helpers.receivers_local(mesh, dim, receivers.receiver_locations)
    outfile = helpers.create_output_file("SSPRKMODJustNoPML_G"+str(G)+".pvd", comm, source_num)

    t = 0.0

    cutoff = freq_bands[freq_index] if "inversion" in model else None
    RW = FullRickerWavelet(dt, tf, freq, amp=amp, cutoff=cutoff)

    excitation = excitations[source_num]
    if model['acquisition']['source_mesh_point']== False:
        ricker = Constant(0)
        f = excitation * ricker
        ricker.assign(RW[0])
    if model['acquisition']['source_mesh_point']== True:
        f = Function(V)
        dof = model['acquisition']["source_point_dof"]
        f.dat.data[dof] = RW[0]

    # -------------------------------------------------------
    m1 = (dwdt_trial) * q1 * dx(rule=qr_x)
    a = c * c * dot(grad(p_n), grad(q1)) * dx(rule=qr_x)  # explicit

    FF = m1 + a - f * q1 * dx(rule=qr_x)

    X = Function(W)
    B = Function(W)

    # DOF FROM RK substitution
    FF += dpdt_trial*q2*dx(rule=qr_x) - w_n*q2*dx(rule=qr_x)
    
    #lhs_ = lhs(FF)
    #rhs_ = rhs(FF)
    lhs_ = (dwdt_trial)*q1*dx(rule=qr_x) + dpdt_trial*q2*dx(rule=qr_x)
    rhs_ = - c*c*dot(grad(p_n), grad(q1))*dx(rule=qr_x) + f*q1*dx(rule=qr_x) + w_n*q2*dx(rule=qr_x)

    A = assemble(lhs_, mat_type="matfree")
    solver = LinearSolver(A, solver_parameters=params)

    usol = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
    usol_recv = []
    saveIT = 0

    X0 = Function(W)
    w0, p0= X0.split()

    X1 = Function(W)
    X2 = Function(W)
    X3 = Function(W)

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    for IT in range(nt):

        if IT < dstep:
            if model['acquisition']['source_mesh_point']== False:
                ricker.assign(RW[IT])
            elif model['acquisition']['source_mesh_point']== True:
                f.dat.data[dof] = RW[IT]
        elif IT == dstep:
            if model['acquisition']['source_mesh_point']== False:
                ricker.assign(0.0)
            elif model['acquisition']['source_mesh_point']== True:
                f.dat.data[dof] = 0.0

        # AX=B --> solve for X = B/Aˆ-1
        B = assemble(rhs_, tensor=B)
        #assembly_callable()


        ## FISRT STEP
        solver.solve(X, B)
        X1.assign(X)
        x1w, x1p= X1.split()

        w_n.assign( w0  + dt*x1w)
        p_n.assign( p0  + dt*x1p)

        ## SECOND STEP
        #assembly_callable()
        if IT < dstep-1:
            if model['acquisition']['source_mesh_point']== False:
                ricker.assign(RW[IT+1])
            elif model['acquisition']['source_mesh_point']== True:
                f.dat.data[dof] = RW[IT+1]
        B = assemble(rhs_, tensor=B)
        solver.solve(X, B)
        X2.assign(X)
        x2w, x2p = X2.split()

        w_n.assign( 0.75*w0  + 0.25*(w_n  + dt*x2w) )
        p_n.assign( 0.75*p0  + 0.25*(p_n  + dt*x2p) )

        ## THIRD STEP
        #assembly_callable()
        if IT < dstep-2:
            if model['acquisition']['source_mesh_point']== False:
                ricker.assign(3./4.*RW[IT] + 1./4.*RW[IT+2] )
            elif model['acquisition']['source_mesh_point']== True:
                f.dat.data[dof] = RW[IT]
        B = assemble(rhs_, tensor=B)
        solver.solve(X, B)
        X3.assign(X)
        x3w, x3p= X3.split()

        w_n.assign( (1./3.)*w0  + 0.25*(w_n  + dt*x2w) )
        p_n.assign( (1./3.)*p0  + 0.25*(p_n  + dt*x2p) )

        ## UPDATING
        w0.assign( w_n)
        p0.assign( p_n)

        usol_recv.append(receivers.interpolate(p_n.dat.data_ro_with_halos[:], is_local))

        if IT % fspool == 0:
            usol[saveIT].assign(p_n)
            saveIT += 1

        if IT % nspool == 0:
            assert (
                 norm(p_n) < 1
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                outfile.write(p_n, time=t, name="Pressure")
            helpers.display_progress(comm, t)

        t = IT * float(dt)

    usol_recv = helpers.fill(usol_recv, is_local, nt, receivers.num_receivers)
    usol_recv = utils.communicate(usol_recv, comm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return usol, usol_recv
