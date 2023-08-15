import math
from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..io import ensemble_forward
from ..pml import damping
from . import helpers
from .CG_acoustic import AcousticWave
from ..io.basicio import ensemble_propagator

# Note this turns off non-fatal warnings
set_log_level(ERROR)


def dampingfunctions(
    model,
    V,
    dimension,
    x,
    x1,
    x2,
    a_pml,
    z,
    z1,
    z2,
    c_pml,
    y=None,
    y1=None,
    y2=None,
    b_pml=None,
):

    ps = model["exponent"]  # polynomial scaling
    cmax = model["cmax"]  # maximum acoustic wave velocity
    R = model["R"]  # theoretical reclection coefficient

    bar_sigma = ((3.0 * cmax) / (2.0 * a_pml)) * math.log10(1.0 / R)
    aux1 = Function(V)
    aux2 = Function(V)

    # Sigma X
    sigma_max_x = bar_sigma  # Max damping
    aux1.interpolate(
        conditional(
            And((x >= x1 - a_pml), x < x1),
            ((abs(x - x1) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    aux2.interpolate(
        conditional(
            And(x > x2, (x <= x2 + a_pml)),
            ((abs(x - x2) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    sigma_x = Function(V, name="sigma_x").interpolate(aux1 + aux2)

    # Sigma Z
    tol_z = 1.000001
    sigma_max_z = bar_sigma  # Max damping
    aux1.interpolate(
        conditional(
            And(z < z2, (z >= z2 - tol_z * c_pml)),
            ((abs(z - z2) ** (ps)) / (c_pml ** (ps))) * sigma_max_z,
            0.0,
        )
    )

    sigma_z = Function(V, name="sigma_z").interpolate(aux1)

    # sgm_x = File("pmlField/sigma_x.pvd")  # , target_degree=1, target_continuity=H1
    # sgm_x.write(sigma_x)
    # sgm_z = File("pmlField/sigma_z.pvd")
    # sgm_z.write(sigma_z)

    if dimension == 2:

        return (sigma_x, sigma_z)

    elif dimension == 3:
        # Sigma Y
        sigma_max_y = bar_sigma  # Max damping
        aux1.interpolate(
            conditional(
                And((y >= y1 - b_pml), y < y1),
                ((abs(y - y1) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(y > y2, (y <= y2 + b_pml)),
                ((abs(y - y2) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        sigma_y = Function(V, name="sigma_y").interpolate(aux1 + aux2)
        # sgm_y = File("pmlField/sigma_y.pvd")
        # sgm_y.write(sigma_y)

        return (sigma_x, sigma_y, sigma_z)


def dampingmatrices_2D(sigma_x, sigma_y):
    """Damping matrices for a two-dimensional problem"""
    Gamma_1 = as_tensor([[sigma_x, 0.0], [0.0, sigma_y]])
    Gamma_2 = as_tensor([[sigma_x - sigma_y, 0.0], [0.0, sigma_y - sigma_x]])

    return (Gamma_1, Gamma_2)


def matrices_3D(sigma_x, sigma_y, sigma_z):
    """Damping matrices for a three-dimensional problem"""
    Gamma_1 = as_tensor([[sigma_x, 0.0, 0.0], [0.0, sigma_y, 0.0], [0.0, 0.0, sigma_z]])
    Gamma_2 = as_tensor(
        [
            [sigma_x - sigma_y - sigma_z, 0.0, 0.0],
            [0.0, sigma_y - sigma_x - sigma_z, 0.0],
            [0.0, 0.0, sigma_z - sigma_x - sigma_y],
        ]
    )
    Gamma_3 = as_tensor(
        [
            [sigma_y * sigma_z, 0.0, 0.0],
            [0.0, sigma_x * sigma_z, 0.0],
            [0.0, 0.0, sigma_x * sigma_y],
        ]
    )

    return (Gamma_1, Gamma_2, Gamma_3)


class temp_pml(AcousticWave):
    def forward_solve(self):
        """Solves the forward problem.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self._get_initial_velocity_model()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

    def matrix_building(self):
        degree = self.degree
        dt = self.dt
        mesh = self.mesh
        c = self.c

        params = {"ksp_type": "preonly", "pc_type": "jacobi"}

        element = FiniteElement("KMV", mesh.ufl_cell(), degree, variant="KMV")

        V = FunctionSpace(mesh, element)

        qr_x, qr_s, _ = quadrature.quadrature_rules(V)

        z, x = SpatialCoordinate(mesh)

        Z = VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        W = V * Z
        u, pp = TrialFunctions(W)
        v, qq = TestFunctions(W)

        u_np1, pp_np1 = Function(W).split()
        u_n, pp_n = Function(W).split()
        u_nm1, pp_nm1 = Function(W).split()

        self.u_n = u_n
        self.pp_n = pp_n
        self.u_nm1 = u_nm1
        self.pp_nm1 = pp_nm1

        # sigma_x, sigma_z = dampingfunctions(
        #     pml_dict, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
        # )
        sigma_x, sigma_z = damping.functions(self)
        Gamma_1, Gamma_2 = dampingmatrices_2D(sigma_z, sigma_x)
        pml1 = (
            (sigma_x + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v
            * dx(scheme=qr_x)
        )

        # typical CG FEM in 2d/3d

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=qr_x)
        a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=qr_x)  # explicit

        nf = c * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

        FF = m1 + a + nf

        X = Function(W)
        B = Function(W)

        pml2 = sigma_x * sigma_z * u_n * v * dx(scheme=qr_x)
        pml3 = inner(pp_n, grad(v)) * dx(scheme=qr_x)
        FF += pml1 + pml2 + pml3
        # -------------------------------------------------------
        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(scheme=qr_x)
        mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(scheme=qr_x)
        dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dx(scheme=qr_x)
        FF += mm1 + mm2 + dd

        lhs_ = lhs(FF)
        rhs_ = rhs(FF)

        A = assemble(lhs_, mat_type="matfree")
        solver = LinearSolver(A, solver_parameters=params)
        self.solver = solver
        self.rhs = rhs_
        self.B = B
        self.X = X

        return

    @ensemble_propagator
    def wave_propagator(self, dt=None, final_time=None, source_num=0):
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
        # degree = self.degree
        # mesh = self.mesh
        # c = self.c


        # params = {"ksp_type": "preonly", "pc_type": "jacobi"}

        # element = FiniteElement("KMV", mesh.ufl_cell(), degree, variant="KMV")


        # qr_x, qr_s, _ = quadrature.quadrature_rules(V)

        # z, x = SpatialCoordinate(mesh)

        # Z = VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        # W = V * Z
        # u, pp = TrialFunctions(W)
        # v, qq = TestFunctions(W)

        # u_np1, pp_np1 = Function(W).split()
        # u_n, pp_n = Function(W).split()
        # u_nm1, pp_nm1 = Function(W).split()

        # # sigma_x, sigma_z = dampingfunctions(
        # #     pml_dict, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
        # # )
        # sigma_x, sigma_z = damping.functions(self)
        # Gamma_1, Gamma_2 = dampingmatrices_2D(sigma_z, sigma_x)
        # pml1 = (
        #     (sigma_x + sigma_z)
        #     * ((u - u_nm1) / Constant(2.0 * dt))
        #     * v
        #     * dx(scheme=qr_x)
        # )

        # # typical CG FEM in 2d/3d


        # # -------------------------------------------------------
        # m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=qr_x)
        # a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=qr_x)  # explicit

        # nf = c * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

        # FF = m1 + a + nf

        # X = Function(W)
        # B = Function(W)

        # pml2 = sigma_x * sigma_z * u_n * v * dx(scheme=qr_x)
        # pml3 = inner(pp_n, grad(v)) * dx(scheme=qr_x)
        # FF += pml1 + pml2 + pml3
        # # -------------------------------------------------------
        # mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(scheme=qr_x)
        # mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(scheme=qr_x)
        # dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dx(scheme=qr_x)
        # FF += mm1 + mm2 + dd
        t = 0.0

        u_n = self.u_n
        pp_n = self.pp_n
        u_nm1 = self.u_nm1
        pp_nm1 = self.pp_nm1

        comm = self.comm
        dt = self.dt
        tf = self.final_time
        nt = int(tf / dt) + 1  # number of timesteps
        V = self.function_space
        wavelet = self.wavelet
        receivers = self.receivers
        excitations = self.sources
        excitations.current_source = source_num
        solver = self.solver
        rhs_ = self.rhs
        B = self.B
        X = self.X

        nspool = self.output_frequency
        fspool = self.gradient_sampling_frequency

        usol = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
        usol_recv = []
        save_step = 0

        assembly_callable = create_assembly_callable(rhs_, tensor=B)

        rhs_forcing = Function(V)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            assembly_callable()
            f = excitations.apply_source(rhs_forcing, wavelet[step])
            B0 = B.sub(0)
            B0 += f
            solver.solve(X, B)

            u_np1, pp_np1 = X.split()

            pp_nm1.assign(pp_n)
            pp_n.assign(pp_np1)

            usol_recv.append(receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

            if step % fspool == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if step % nspool == 0:
                assert (
                    norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if t > 0:
                    helpers.display_progress(comm, t)

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            t = step * float(dt)

        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.number_of_points)
        usol_recv = utils.utils.communicate(usol_recv, comm)

        self.forward_solution = usol
        self.forward_solution_receivers = usol_recv

        return usol, usol_recv
