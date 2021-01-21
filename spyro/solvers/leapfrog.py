from firedrake import *

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..sources import FullRickerWavelet
from . import helpers

__all__ = ["Leapfrog"]


def _unpack_pml(model):
    if model["PML"]:
        Lx = model["mesh"]["Lx"]
        Lz = model["mesh"]["Lz"]
        lx = model["PML"]["lx"]
        lz = model["PML"]["lz"]
        x1 = 0.0
        x2 = Lx
        a_pml = lx
        z1 = 0.0
        z2 = -Lz
        c_pml = lz
        if model["opts"]["dimension"] == 3:
            Ly = model["mesh"]["Ly"]
            ly = model["PML"]["ly"]
            y1 = 0.0
            y2 = Ly
            b_pml = ly
            return Lx, lx, Lz, lz, Ly, ly, x1, x2, a_pml, z1, z2, c_pml, y1, y2, b_pml
        return Lx, lx, Lz, lz, x1, x2, a_pml, z1, z2, c_pml


class Leapfrog:
    def __init__(
        self,
        model,
        mesh,
        comm,
        c,
        excitations,
        receivers,
        source_num=0,
        lp_freq_index=0,
    ):
        self.model = model
        self.mesh = mesh
        self.comm = comm
        self.c = c
        self.dim = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        self.amp = 1
        if "amplitude" in model["acquisition"]:
            self.amp = model["acquisition"]["amplitude"]
        self.freq = model["acquisition"]["frequency"]
        self.lp_freq_index = lp_freq_index
        if "inversion" in model:
            self.freq_bands = model["inversion"]["freq_bands"]
        self.dt = model["timeaxis"]["dt"]
        self.tf = model["timeaxis"]["tf"]
        self.delay = model["acquisition"]["delay"]
        self.nspool = model["timeaxis"]["nspool"]
        self.fspool = model["timeaxis"]["fspool"]
        self.PML = model["PML"]["status"]

        if self.method == "KMV":
            self.params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        elif self.method == "CG":
            self.params = {"ksp_type": "cg", "pc_type": "jacobi"}
        else:
            raise ValueError("method is not yet supported")

        self.source_num = source_num

        self._nt = int(self.tf / self.dt)  # number of timesteps
        self._dstep = int(self.delay / self.dt)  # number of timesteps with source

        element = space.FE_method(self.mesh, self.method, self.degree)

        self.V = FunctionSpace(self.mesh, element)

        cutoff = self.freq_bands[lp_freq_index] if "inversion" in model else None
        RW = FullRickerWavelet(self.dt, self.tf, self.freq, amp=self.amp, cutoff=cutoff)

        self.ricker = Constant(0)
        self.receivers = receivers
        self.excitations = excitations

        self._build_variational_form()

    def _build_variational_form(self):

        sd = self.dim
        if sd == 2:
            z, x = SpatialCoordinate(self.mesh)
        elif sd == 3:
            z, x, y = SpatialCoordinate(self.mesh)

        qr_x, qr_s, _ = quadrature.quadrature_rules(self.V)

        if self.PML:
            Z = VectorFunctionSpace(self.V.ufl_domain(), self.V.ufl_element())
            if sd == 2:
                W = self.V * Z
                u, pp = TrialFunctions(W)
                v, qq = TestFunctions(W)

                self.u_np1, self.pp_np1 = Function(W).split()
                self.u_n, self.pp_n = Function(W).split()
                self.u_nm1, self.pp_nm1 = Function(W).split()

            elif sd == 3:
                W = self.V * self.V * Z
                u, psi, pp = TrialFunctions(W)
                v, phi, qq = TestFunctions(W)

                self.u_np1, self.psi_np1, self.pp_np1 = Function(W).split()
                self.u_n, self.psi_n, self.pp_n = Function(W).split()
                self.u_nm1, self.psi_nm1, self.pp_nm1 = Function(W).split()

            if sd == 2:

                Lx, lx, Lz, lz, x1, x2, a_pml, z1, z2, c_pml = _unpack_pml(self.model)

                (sigma_x, sigma_z) = damping.functions(
                    self.model, self.V, sd, x, x1, x2, a_pml, z, z1, z2, c_pml
                )
                (Gamma_1, Gamma_2) = damping.matrices_2D(sigma_z, sigma_x)
                pml1 = (
                    (sigma_x + sigma_z)
                    * ((u - self.u_nm1) / Constant(2.0 * self.dt))
                    * v
                    * dx(rule=qr_x)
                )
            elif sd == 3:
                (
                    Lx,
                    lx,
                    Lz,
                    lz,
                    Ly,
                    ly,
                    x1,
                    x2,
                    a_pml,
                    z1,
                    z2,
                    c_pml,
                    y1,
                    y2,
                    b_pml,
                ) = _unpack_pml(self.model)
                sigma_x, sigma_y, sigma_z = damping.functions(
                    self.model,
                    self.V,
                    sd,
                    x,
                    x1,
                    x2,
                    a_pml,
                    z,
                    z1,
                    z2,
                    c_pml,
                    y,
                    y1,
                    y2,
                    b_pml,
                )
                Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(
                    sigma_x, sigma_y, sigma_z
                )

        # typical CG in N-d with no PML
        else:
            u = TrialFunction(self.V)
            v = TestFunction(self.V)

            self.u_nm1 = Function(self.V)
            self.u_n = Function(self.V)
            self.u_np1 = Function(self.V)

        self.is_local = helpers.receivers_local(
            self.mesh, sd, self.receivers.receiver_locations
        )

        cutoff = (
            self.freq_bands[self.lp_freq_index] if "inversion" in self.model else None
        )
        self.RW = FullRickerWavelet(
            self.dt, self.tf, self.freq, amp=self.amp, cutoff=cutoff
        )

        excitation = self.excitations[self.source_num]
        f = excitation * self.ricker
        self.ricker.assign(self.RW[0])
        # -------------------------------------------------------
        m1 = (
            ((u - 2.0 * self.u_n + self.u_nm1) / Constant(self.dt ** 2))
            * v
            * dx(rule=qr_x)
        )
        a = self.c * self.c * dot(grad(self.u_n), grad(v)) * dx(rule=qr_x)  # explicit

        if self.model["PML"]["outer_bc"] == "non-reflective":
            nf = self.c * ((self.u_n - self.u_nm1) / self.dt) * v * ds(rule=qr_s)
        else:
            nf = 0

        FF = m1 + a + nf - f * v * dx(rule=qr_x)

        if self.PML:
            X = Function(W)
            B = Function(W)

            if sd == 2:
                pml2 = sigma_x * sigma_z * self.u_n * v * dx(rule=qr_x)
                pml3 = inner(self.pp_n, grad(v)) * dx(rule=qr_x)
                FF += pml1 + pml2 + pml3
                # -------------------------------------------------------
                mm1 = (dot((pp - self.pp_n), qq) / Constant(self.dt)) * dx(rule=qr_x)
                mm2 = inner(dot(Gamma_1, self.pp_n), qq) * dx(rule=qr_x)
                dd = inner(grad(self.u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
                FF += mm1 + mm2 + dd
            elif sd == 3:
                pml1 = (
                    (sigma_x + sigma_y + sigma_z)
                    * ((u - self.u_n) / Constant(dt))
                    * v
                    * dx(rule=qr_x)
                )
                pml2 = (
                    (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
                    * self.u_n
                    * v
                    * dx(rule=qr_x)
                )
                pml3 = (sigma_x * sigma_y * sigma_z) * self.psi_n * v * dx(rule=qr_x)
                pml4 = inner(self.pp_n, grad(v)) * dx(rule=qr_x)

                FF += pml1 + pml2 + pml3 + pml4
                # -------------------------------------------------------
                mm1 = (dot((pp - self.pp_n), qq) / Constant(self.dt)) * dx(rule=qr_x)
                mm2 = inner(dot(Gamma_1, self.pp_n), qq) * dx(rule=qr_x)
                dd1 = inner(grad(self.u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
                dd2 = -inner(grad(self.psi_n), dot(Gamma_3, qq)) * dx(rule=qr_x)

                FF += mm1 + mm2 + dd1 + dd2
                # -------------------------------------------------------
                mmm1 = (dot((psi - self.psi_n), phi) / Constant(self.dt)) * dx(
                    rule=qr_x
                )
                uuu1 = (-self.u_n * phi) * dx(rule=qr_x)

                FF += mm1 + uuu1
        else:
            X = Function(self.V)
            B = Function(self.V)

        lhs_ = lhs(FF)
        rhs_ = rhs(FF)

        A = assemble(lhs_, mat_type="matfree")

        self.solver = LinearSolver(A, solver_parameters=self.params)
        self.rhs_ = rhs_
        self.X = X
        self.B = B

    def _set_initial_conditions(self):
        sd = self.dim
        if self.PML:
            if sd == 2:
                self.u_np1.assign(0.0), self.pp_np1.assign(0.0)
                self.u_n.assign(0.0), self.pp_n.assign(0.0)
                self.u_nm1.assign(0.0), self.pp_nm1.assign(0.0)
            elif dim == 3:
                self.u_np1.assign(0.0), self.psi_np1.assign(0.0), self.pp_np1.assign(
                    0.0
                )
                self.u_n.assign(0.0), self.psi_n.assign(0.0), self.pp_n.assign(0.0)
                self.u_nm1.assign(0.0), self.psi_nm1.assign(0.0), self.pp_nm1.assign(
                    0.0
                )
        else:
            self.X.assign(0.0)
            self.B.assign(0.0)

    def timestep(self, write=False):

        sd = self.dim

        if write:
            outfile = helpers.create_output_file(
                "Leapfrog.pvd", self.comm, self.source_num
            )

        usol = [
            Function(self.V, name="pressure")
            for t in range(self._nt)
            if t % self.fspool == 0
        ]
        usol_recv = []
        save_step = 0

        t = 0.0

        # zero out all terms
        self._set_initial_conditions()

        for step in range(self._nt):

            if step < self._dstep:
                self.ricker.assign(self.RW[step])
            elif step == self._dstep:
                self.ricker.assign(0.0)

            # AX=B --> solve for X = B/Aˆ-1
            self.B = assemble(self.rhs_, tensor=self.B)
            self.solver.solve(self.X, self.B)
            if self.PML:
                if sd == 2:
                    self.u_np1, self.pp_np1 = self.X.split()
                elif dim == 3:
                    self.u_np1, self.psi_np1, self.pp_np1 = self.X.split()

                    self.psi_nm1.assign(self.psi_n)
                    self.psi_n.assign(self.psi_np1)

                self.pp_nm1.assign(self.pp_n)
                self.pp_n.assign(self.pp_np1)
            else:
                self.u_np1.assign(self.X)

            self.u_nm1.assign(self.u_n)
            self.u_n.assign(self.u_np1)

            usol_recv.append(
                self.receivers.interpolate(
                    self.u_n.dat.data_ro_with_halos[:], self.is_local
                )
            )

            if step % self.fspool == 0:
                usol[save_step].assign(self.u_n)
                save_step += 1

            if step % self.nspool == 0:
                if write:
                    outfile.write(self.u_n, time=t)
                helpers.display_progress(self.comm, t)

            t = step * float(self.dt)

        usol_recv = helpers.fill(
            usol_recv, self.is_local, self._nt, self.receivers.num_receivers
        )
        usol_recv = utils.communicate(usol_recv, self.comm)

        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            print(
                "---------------------------------------------------------------",
                flush=True,
            )

        return usol, usol_recv
