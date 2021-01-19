from __future__ import print_function

import numpy as np
from firedrake import *
from scipy.sparse import csc_matrix

from .. import io
from ..domains import quadrature, space
from ..pml import damping
from ..sources import delta_expr, delta_expr_3d
from . import helpers

set_log_level(ERROR)

__all__ = ["LeapfrogAdjoint"]


def _adjoint_update_rhs(rhs_forcing, excitations, residual, IT, is_local):
    """Builds assembled forcing function f for adjoint for a given time_step
    given a number of receivers
    """
    recs = [recv for recv in range(excitations.shape[1]) if is_local[recv]]
    rhs_forcing.dat.data[:] = excitations[:, recs].dot(residual[IT][recs])

    return rhs_forcing


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


class LeapfrogAdjoint:
    def __init__(self, model, mesh, comm, c, guess, residual):
        self.model = model
        self.mesh = mesh
        self.comm = comm
        self.c = c
        self.guess = guess
        self.residual = residual
        self.dim = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        if "inversion" in model:
            self.freq_bands = model["inversion"]["freq_bands"]
        self.dt = model["timeaxis"]["dt"]
        self.tf = model["timeaxis"]["tf"]
        self.delay = model["acquisition"]["delay"]
        self.nspool = model["timeaxis"]["nspool"]
        self.fspool = model["timeaxis"]["fspool"]
        self.numrecs = model["acquisition"]["num_receivers"]
        self.PML = model["PML"]["status"]

        if self.method == "KMV":
            self.params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        elif self.method == "CG":
            self.params = {"ksp_type": "cg", "pc_type": "jacobi"}
        else:
            raise ValueError("method is not yet supported")

        self._nt = int(self.tf / self.dt)  # number of timesteps

        element = space.FE_method(self.mesh, self.method, self.degree)

        self.V = FunctionSpace(self.mesh, element)

        self._build_variational_form()

    def _build_variational_form(self):

        sd = self.dim
        # Prepare receiver forcing terms
        if sd == 2:
            z, x = SpatialCoordinate(self.mesh)
            receiver = Constant([0, 0])
            delta = Interpolator(delta_expr(receiver, z, x), self.V)
        elif sd == 3:
            z, x, y = SpatialCoordinate(self.mesh)
            receiver = Constant([0, 0, 0])
            delta = Interpolator(delta_expr_3d(receiver, z, x, y), self.V)

        qr_x, qr_s, _ = quadrature.quadrature_rules(self.V)

        receiver_locations = self.model["acquisition"]["receiver_locations"]

        if sd == 2:
            self.is_local = [
                self.mesh.locate_cell([z, x]) for z, x in receiver_locations
            ]
        elif sd == 3:
            self.is_local = [
                self.mesh.locate_cell([z, x, y]) for z, x, y in receiver_locations
            ]

        self.dJdC = Function(self.V)

        # receivers are forced through sparse matrix vec multiplication
        self.sparse_excitations = csc_matrix((len(self.dJdC.dat.data), self.numrecs))
        for r, x0 in enumerate(receiver_locations):
            receiver.assign(x0)
            exct = delta.interpolate().dat.data_ro.copy()
            row = exct.nonzero()[0]
            col = np.repeat(r, len(row))
            sparse_exct = csc_matrix(
                (exct[row], (row, col)), shape=self.sparse_excitations.shape
            )
            self.sparse_excitations += sparse_exct

        # if using the PML
        if self.PML:
            Z = VectorFunctionSpace(self.V.ufl_domain(), self.V.ufl_element())
            if sd == 2:
                W = self.V * Z
                u, pp = TrialFunctions(W)
                v, qq = TestFunctions(W)

                self.u_np1, self.pp_np1 = Function(W).split()
                self.u_n, self.pp_n = Function(W).split()
                self.u_nm1, self.pp_nm1 = Function(W).split()

            elif dim == 3:
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
                    * ((u - self.u_nm1) / (2.0 * Constant(self.dt)))
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
        else:
            u = TrialFunction(self.V)
            v = TestFunction(self.V)

            self.u_nm1 = Function(self.V)
            self.u_n = Function(self.V)
            self.u_np1 = Function(self.V)

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

        FF = m1 + a + nf

        if self.PML:
            X = Function(W)
            B = Function(W)

            if sd == 2:
                pml1 = (
                    (sigma_x + sigma_z) * ((u - self.u_n) / self.dt) * v * dx(rule=qr_x)
                )
                pml2 = sigma_x * sigma_z * self.u_n * v * dx(rule=qr_x)
                pml3 = inner(grad(v), dot(Gamma_2, self.pp_n)) * dx(rule=qr_x)

                FF += pml1 + pml2 + pml3
                # -------------------------------------------------------
                mm1 = (dot((pp - self.pp_n), qq) / Constant(self.dt)) * dx(rule=qr_x)
                mm2 = inner(dot(Gamma_1, self.pp_n), qq) * dx(rule=qr_x)
                dd = inner(qq, grad(self.u_n)) * dx(rule=qr_x)

                FF += mm1 + mm2 + dd
            elif sd == 3:
                pml1 = (
                    (sigma_x + sigma_y + sigma_z)
                    * ((u - self.u_n) / self.dt)
                    * v
                    * dx(rule=qr_x)
                )
                pml2 = (
                    (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
                    * self.u_n
                    * v
                    * dx(rule=qr_x)
                )
                dd1 = inner(grad(v), dot(Gamma_2, self.pp_n)) * dx(rule=qr_x)

                FF += pml1 + pml2 + dd1
                # -------------------------------------------------------
                mm1 = (dot((pp - self.pp_n), qq) / self.dt) * dx(rule=qr_x)
                mm2 = inner(dot(Gamma_1, self.pp_n), qq) * dx(rule=qr_x)
                pml4 = inner(qq, grad(self.u_n)) * dx(rule=qr_x)

                FF += mm1 + mm2 + pml4
                # -------------------------------------------------------
                pml3 = (sigma_x * sigma_y * sigma_z) * phi * self.u_n * dx(rule=qr_x)
                mmm1 = (dot((psi - self.psi_n), phi) / self.dt) * dx(rule=qr_x)
                uuu1 = (-self.u_n * phi) * dx(rule=qr_x)

                FF += mm1 + uuu1 + pml3
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

        # Define gradient problem
        g_u = TrialFunction(self.V)
        g_v = TestFunction(self.V)

        mgrad = g_u * g_v * dx(rule=qr_x)

        self.uuadj = Function(self.V)  # auxiliarly function for the gradient compt.
        self.uufor = Function(self.V)  # auxiliarly function for the gradient compt.

        if self.PML:
            self.ppadj = Function(Z)  # auxiliarly function for the gradient compt.
            self.ppfor = Function(Z)  # auxiliarly function for the gradient compt.

            ffG = (
                2.0
                * self.c
                * Constant(self.dt)
                * (
                    dot(grad(self.uuadj), grad(self.uufor))
                    + inner(grad(self.uufor), dot(Gamma_2, self.ppadj))
                )
                * g_v
                * dx(rule=qr_x)
            )
        else:
            ffG = (
                2.0
                * self.c
                * Constant(self.dt)
                * dot(grad(self.uuadj), grad(self.uufor))
                * g_v
                * dx(rule=qr_x)
            )

        G = mgrad - ffG
        lhsG, rhsG = lhs(G), rhs(G)

        gradi = Function(self.V)
        grad_prob = LinearVariationalProblem(lhsG, rhsG, gradi)
        if self.method == "KMV":
            self.grad_solv = LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                    "mat_type": "matfree",
                },
            )
        elif self.method == "CG":
            self.grad_solv = LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "mat_type": "matfree",
                },
            )

    def timestep(self):

        # outfile = helpers.create_output_file("Leapfrog_adjoint.pvd", self.comm, source_num)

        t = 0.0

        gradi = Function(self.V)

        rhs_forcing = Function(self.V)  # forcing term
        for step in range(self._nt - 1, 0, -1):
            t = step * float(self.dt)

            # Solver - main equation - (I)
            self.B = assemble(self.rhs_, tensor=self.B)
            f = _adjoint_update_rhs(
                rhs_forcing, self.sparse_excitations, self.residual, step, self.is_local
            )
            # add forcing term to solve scalar pressure
            self.B.sub(0).dat.data[:] += f.dat.data[:]

            # AX=B --> solve for X = B/Aˆ-1
            self.solver.solve(self.X, self.B)
            if self.PML:
                if sd == 2:
                    self.u_np1, self.pp_np1 = X.split()
                elif sd == 3:
                    self.u_np1, self.psi_np1, self.pp_np1 = X.split()

                    self.psi_nm1.assign(self.psi_n)
                    self.psi_n.assign(self.psi_np1)

                self.pp_nm1.assign(self.pp_n)
                self.pp_n.assign(self.pp_np1)
            else:
                self.u_np1.assign(self.X)

            self.u_nm1.assign(self.u_n)
            self.u_n.assign(self.u_np1)

            # compute the gradient increment
            self.uuadj.assign(self.u_n)

            # only compute for snaps that were saved
            if step % self.fspool == 0:
                gradi.assign = 0.0
                self.uufor.assign(self.guess.pop())

                self.grad_solv.solve()
                self.dJdC += gradi

            if step % self.nspool == 0:
                # outfile.write(self.u_n, time=t)
                helpers.display_progress(self.comm, t)

        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            print(
                "---------------------------------------------------------------",
                flush=True,
            )

        return self.dJdC
