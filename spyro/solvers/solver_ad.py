import firedrake as fire
import numpy as np
from mpi4py import MPI
from .. import utils
from ..domains import quadrature
from ..pml import damping, aux_equations
from . import helpers
from .. import tools
import copy
# Note this turns off non-fatal warnings
fire.set_log_level(fire.ERROR)


class solver_ad():
    def __init__(
            self, model, mesh, V, source_num=0, solver="fwd"
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
        """
        self.mesh = mesh
        self.model = model
        self.source_num = source_num
        self.V = V
        self.solver = solver
        self.tolerance = 0.0000001
        rec_loc = self.model["acquisition"]["receiver_locations"]
        self.h_min = tools.min_equilateral_distance(
                                            self.mesh,
                                            self.V,
                                            rec_loc)
        self.h_min = fire.COMM_WORLD.allreduce(self.h_min, op=MPI.MIN)
       
    def wave_propagate(
                        self, comm, c, excitation, receivers,
                        wavelet, output=False, **kwargs
                        ):
        """Forward and adjoint wave equation solver.

        Parameters
        ----------
        comm: Firedrake.ensemble_communicator
            The MPI communicator for parallelism
        c: Firedrake.Function
            The velocity model interpolated onto the mesh.
        excitations: A list Firedrake.Functions
        receivers: A :class:`spyro.Receivers` object.
            Contains the receiver locations and sparse 
            interpolation method fire.
        wavelet: array-like
            Time series data that's injected at the source location.
        output: `boolean`, optional
            Whether or not to write results to pvd files.

        Returns
        -------
        usol: list of Firedrake.Functions
            The full field solution at `fspool` timesteps
        usol_recv: array-like
            The solution interpolated to the receivers at all timesteps
        J0: Cost function associated to a single source 
        dJ: Gradient field associate to a single source when the
            implemented adjoint is employed
        misfit: The misfit function
        """
        model = self.model
        method = model["opts"]["method"]
        dim = model["opts"]["dimension"]
        dt = model["timeaxis"]["dt"]
        tf = model["timeaxis"]["tf"]
        nt = int(tf / dt)  # number of timesteps
        nspool = model["timeaxis"]["nspool"]
        fspool = model["timeaxis"]["fspool"]
        bc = model["BCs"]["status"]
        V = self.V
        
        # kwargs
        save_misfit = kwargs.get("save_misfit")
        calc_functional = kwargs.get("calc_functional")
        save_rec_data = kwargs.get("save_rec_data")
        p_true_rec = kwargs.get("true_rec")
        save_p = kwargs.get("save_p")

        excitation.current_source = self.source_num
        
        params = self.set_params(method)
        qr_x, qr_s, _ = quadrature.quadrature_rules(V)
        if output:
            outfile = helpers.create_output_file(
                                        "forward.pvd", comm, self.source_num
                                        )
        if dim == 2:
            z, x = fire.SpatialCoordinate(self.mesh)
            if bc:
                sigma_x, sigma_z = self.set_bc_params(dim, x, z)                          
        elif dim == 3:
            z, x, y = fire.SpatialCoordinate(self.mesh)
            if bc:
                sigma_x, sigma_y, sigma_z = self.set_bc_params(dim, x, z, y)   
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)  # Test Function
        f = fire.Function(V, name="f")
        X = fire.Function(V, name="X")
        u_n = fire.Function(V, name="u_n")      # n
        u_nm1 = fire.Function(V, name="u_nm1")      # n-1

        du2_dt2 = ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt ** 2))
        t_term = du2_dt2 * v * fire.dx(rule=qr_x)
        l_term = c * c * fire.dot(fire.grad(u_n), fire.grad(v)) * fire.dx(rule=qr_x)
        f_term = f * v * fire.dx(rule=qr_x)
        nf = 0
        if model["BCs"]["outer_bc"] == "non-reflective":
            nf = c * ((u_n - u_nm1) / dt) * v * fire.ds(rule=qr_s)    
        FF = t_term + l_term - f_term + nf
        
        if bc:
            if dim == 2:
                FF += (
                        (sigma_x + sigma_z)
                        * ((u - u_nm1) / fire.Constant(2.0 * dt))
                        * v * fire.dx(rule=qr_x)
                    )

                if model["BCs"]["method"] == 'PML':
                    sigma = [sigma_x, sigma_z]
                    aux_pml = aux_equations.set_pml_aux_eq
                    solver0, pp_v, X0 = aux_pml(V, sigma, u_n, v, FF, c, dt,
                                                qr_x, dim, params
                                                )                      
                    pp_n, pp_nm1, pp_np1 = pp_v[0], pp_v[1], pp_v[2]

            if dim == 3:
                FF += (
                        (sigma_x + sigma_y + sigma_z)
                        * ((u - u_n) / fire.Constant(dt))
                        * v * fire.dx(rule=qr_x)
                      )

                if model["BCs"]["method"] == 'PML':
                    sigma = [sigma_x, sigma_z, sigma_y]
                    solver0, solver1, pp_v, psi_v, X1 = aux_pml(
                                                                V, sigma, u_n,
                                                                v, FF, c, dt,
                                                                qr_x, dim,
                                                                params
                                                                )
                    pp_n, pp_nm1, pp_np1 = pp_v[0], pp_v[1], pp_v[2] 
                    psi_n, psi_nm1, psi_np1 = psi_v[0], psi_v[1], psi_v[2] 
          
        lhs_ = fire.lhs(FF)
        rhs_ = fire.rhs(FF)

        problem = fire.LinearVariationalProblem(lhs_, rhs_, X)
        solver = fire.LinearVariationalSolver(
                                    problem, solver_parameters=params
                                    )
        usol_recv = []
        usol = [] 
        misfit = []
        save_step = 0
        # usol = [fire.Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
        if self.solver == "bwd":
            misfit = kwargs.get("misfit")
            dJ = fire.Function(V, name="gradient")
            grad_solver, uuadj, uufor, gradi = self.set_grad_solver(
                                                c, qr_x, method
                                                )
        else:
            interpolator, P = receivers.vertex_only_mesh_interpolator(u_nm1)
            receivers_local_index = receivers.local_receiver_id()
        
        J0 = 0.0
        for step in range(nt):
            if self.solver == "bwd":
                fn = fire.Function(V)
                excitation.apply_receivers_as_source(
                                    fn, misfit, nt-1-step
                                    )
                f.assign(fn)
            else:
                # f_temp = fire.Function(V)
                # f.dat.data[:] = 1.0
                excitation.apply_source(
                        f, wavelet[step]/(self.h_min * self.h_min))
                # f.interpolate(f_temp)

            solver.solve()
            u_nm1.assign(u_n)
            u_n.assign(X)

            if self.solver == "fwd":
                rec = fire.Function(P, name="rec")
                interpolator.interpolate(output=rec)
                if save_rec_data:
                    usol_recv.append(rec.vector().gather())
                if calc_functional:
                    J, misfit_t = utils.compute_functional_ad(
                            rec, p_true_rec[step], P,
                            comm, receivers_local_index)
                    J0 += J
                    if save_misfit:
                        misfit.apppend(misfit_t)

            if bc and model["BCs"]["method"] == 'PML':
                solver0.solve()
                pp_np1.assign(X0)
                pp_nm1.assign(pp_n)
                pp_n.assign(pp_np1)
                if dim == 3:
                    solver1.solve()
                    psi_np1.assign(X1)
                    psi_nm1.assign(psi_n)
                    psi_n.assign(psi_np1)

            if self.solver == "bwd":
                guess = kwargs.get("p_guess")
                uuadj.assign(X)
                uufor.dat.data[:] = guess[len(guess)-1]
                del guess[len(guess)-1]
                grad_solver.solve()
                dJ += gradi

            if save_p and step % fspool == 0:
                # usol[save_step].assign(X)
                usol.append(copy.deepcopy(X.dat.data[:]))
                save_step += 1
            if step % nspool == 0:
                time = step*dt
                # if aut_dif:
                #     from firedrake_adjoint import stop_annotating
                #     with stop_annotating():
                #         helpers.verify_stability(u_n)
                # else:
                #     helpers.verify_stability(u_n)
                
                if output:
                    outfile.write(u_n, time=time, name="Pressure")
                    helpers.display_progress(comm, time)
        
        out = []
        if save_p:
            out.append(usol)
        if calc_functional:
            out.append(J0)
        if save_misfit:
            out.append(misfit)
        if save_rec_data:
            out.append(np.asarray(usol_recv))
        if self.solver == "bwd":
            out.append(dJ)
        return out
        
    def set_params(self, method):
        if method == "KMV":
            params = {
                    "mat_type": "matfree", "ksp_type": 
                    "preonly", "pc_type": "jacobi"
                    }
        elif (
            method == "CG"
            and fire.mesh.ufl_cell() != fire.quadrilateral
            and fire.mesh.ufl_cell() != fire.hexahedron
        ):
            params = {
                    "mat_type": "matfree", "ksp_type": "cg",
                    "pc_type": "jacobi"
                    }
        elif method == "CG" and (
            fire.mesh.ufl_cell() == fire.quadrilateral
            or fire.mesh.ufl_cell() == fire.hexahedron
        ):
            params = {
                    "mat_type": "matfree", "ksp_type": "preonly",
                    "pc_type": "jacobi"
                    }
        else:
            raise ValueError("method is not yet supported")  
        return params
        
    def set_bc_params(self, dim, x, z, y=None):
        model = self.model
        Lx = model["mesh"]["Lx"]
        Lz = model["mesh"]["Lz"]
        lx = model["BCs"]["lx"]
        lz = model["BCs"]["lz"]
        x1 = 0.0
        x2 = Lx
        z1 = 0.0
        z2 = -Lz

        if dim == 2:
            return damping.functions(
                model, self.V, dim, x, x1, x2, lx, z, z1, z2, lz
            )
        elif dim == 3:
            Ly = model["mesh"]["Ly"]
            ly = model["BCs"]["ly"]
            y1 = 0.0
            y2 = Ly

            return damping.functions(
                model, self.V, dim, x, x1, x2, lx, z, z1, z2, lz, y, y1, y2, ly
            )

    def set_grad_solver(self, c, qr_x, method):
        # Define gradient problem
        V = self.V
        m_u = fire.TrialFunction(V)
        m_v = fire.TestFunction(V)
        mgrad = m_u * m_v * fire.dx(rule=qr_x)

        uuadj = fire.Function(V)  # auxiliarly function for the gradient compt.
        uufor = fire.Function(V)  # auxiliarly function for the gradient compt.

        ffG = 2.0 * c * fire.dot(
                    fire.grad(uuadj), fire.grad(uufor)
                    ) * m_v * fire.dx(rule=qr_x)

        G = mgrad - ffG
        lhsG, rhsG = fire.lhs(G), fire.rhs(G)

        gradi = fire.Function(V)
        grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)
        if method == "KMV":
            grad_solver = fire.LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                    "mat_type": "matfree",
                },
            )
        elif method == "CG":
            grad_solver = fire.LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "mat_type": "matfree",
                },
            )
        return grad_solver, uuadj, uufor, gradi