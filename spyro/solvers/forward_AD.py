from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from . import helpers
from ..sources import full_ricker_wavelet, delta_expr

# Note this turns off non-fatal warnings
set_log_level(ERROR)


# @ensemble_forward
def forward(
    model,
    mesh,
    comm,
    c,
    excitations,
    wavelet,
    receivers,
    source_num=0,
    output=False,
    **kwargs
):
    """Secord-order in time fully-explicit scheme
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
    dim    = model["opts"]["dimension"]
    dt     = model["timeaxis"]["dt"]
    tf     = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    delay  = model["acquisition"]["delay"]
    dstep  = int(delay / dt)  # number of timesteps with source
    PML    = model["BCs"]["status"]
    nt     = int(tf / dt)  # number of timesteps
    excitations.current_source = source_num
    params  = set_params(method)
    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = Function(V)
    u_n = Function(V)
    u_np1 = Function(V)

    if output:
        outfile = helpers.create_output_file("forward.pvd", comm, source_num)

    t  = 0.0
    m  = 1/(c*c)
    m1 = m*((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
    a  = dot(grad(u_n), grad(v)) * dx(rule=qr_x)  # explicit
    f  = Function(V)
    nf = 0

    if model["BCs"]["outer_bc"] == "non-reflective":
        nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)
  
    h  = CellSize(mesh)
    FF = m1 + a + nf - (1/(h/degree*h/degree))*f * v * dx(rule=qr_x) 
    X  = Function(V)
    B  = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    problem = LinearVariationalProblem(lhs_, rhs_, X)
    solver  = LinearVariationalSolver(problem, solver_parameters=params)            

    usol_recv = []
    save_step = 0
    
    P            = FunctionSpace(receivers, "DG", 0)
    interpolator = Interpolator(u_np1, P)
    J0           = 0.0 
    
    for step in range(nt):

        excitations.apply_source(f, wavelet[step])
        
        solver.solve()
        u_np1.assign(X)

        rec = Function(P)
        interpolator.interpolate(output=rec)
        
        fwi        = kwargs.get("fwi")
        p_true_rec = kwargs.get("true_rec")
        
        usol_recv.append(rec.dat.data) 

        if fwi:
            J0 += calc_objective_func(
                rec,
                p_true_rec[step],
                step,
                dt,
                P)

        if step % nspool == 0:
            assert (
                norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                outfile.write(u_n, time=t, name="Pressure")
            if t > 0:
                helpers.display_progress(comm, t)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)
    
    if fwi:
        return usol_recv, J0
    else:
        return usol_recv


def calc_objective_func(p_rec,p_true_rec, IT, dt,P):
    true_rec             = Function(P)
    true_rec.dat.data[:] = p_true_rec
    J = 0.5 * assemble(inner(true_rec-p_rec, true_rec-p_rec) * dx)
    return J

def set_params(method):
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
    
    return params
