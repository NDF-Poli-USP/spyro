import firedrake as fire
from mpi4py import MPI
import numpy as np
from sys import float_info, exit
from ...domains.quadrature import quadrature_rules
from ...mpi.spyro_mpi import SpyroEnsemble
from ...utils.error_management import sanitize_num_array, validate_parameter
from ...utils.eval_functions_to_ufl import generate_ufl_functions


# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class Dir_Point_BC(fire.DirichletBC):
    """Class for Eikonal boundary conditions at a point.

    Attributes
    ----------
    nodes : `array`
        Points where the boundary condition is to be applied.
    """

    def __init__(self, V, value, nodes):
        """
        Initialize the Dir_Point_BC class

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space where the boundary condition is applied.
        value : `FiredrakeConstant`
            Value of the boundary condition.
        nodes : `array`
            Points where the boundary condition is to be applied.

        Returns
        -------
        None
        """

        # Calling superclass init and providing a dummy subdomain id
        super(Dir_Point_BC, self).__init__(V, value, 0)

        # Overriding the "nodes" property
        self.nodes = nodes


class Eikonal_Modeling():
    """Class for the Eikonal equation for Linear and Nonlinear analysis.

    Attributes
    ----------
    dimension : `int`
        The spatial dimension of the problem.
    ele_type_eik : `string`, optional
        Finite element type. 'consistent' or 'underintegrated'. Default is 'consistent'.
    f_est : `float`
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.
    degree_eik : `int`
        Finite element order.
    source_locations: `list`of `tuples`
        Source locations as tuples of coordinates.
    tol : `float`, optional
        User solver tolerance. Default is 1e-16.

    Methods
    -------
    nonlinear_eik()
        Assemble the Nonlinear Eikonal with stabilizing term.
    define_int_dom()
        Define the integration domain for the Eikonal equation.
    eikonal_bcs()
        Impose Dirichlet BCs for eikonal equation.
    eikonal_solver()
        Solve the Eikonal equation for model without absorbing layer.
    initial_guess()
        Provide an initial guess for the Eikonal solver.
    linear_eik()
        Assemble the linear Eikonal.
    linear_solution()
        Solve the linear Eikonal equation.
    nonlinear_solution()
        Solve the nonlinear Eikonal equation.
    solver_opts()
        Set the eikonal solver parameters.
    """

    def __init__(self, dimension, source_locations, ele_type_eik='consistent',
                 degree_eik=None, f_est=0.03, tol=1e-16):
        """Initialize the Eikonal_Modeling class.

        Parameters
        ----------
        dimension : `int`
            The spatial dimension of the problem.
        source_locations: `list`of `tuples`
            List of tuples containing all source locations.
        ele_type_eik : `string`, optional
            Finite element type. Options: 'consistent' or 'underintegrated'.
            Default is 'consistent'.
        degree_eik : `int`, optional
            Finite element order for the Eikonal analysis. Default is None.
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.
        tol : `float`, optional
            User solver tolerance. Default is 1e-16.

        Returns
        -------
        None
        """

        # Dimension of the problem
        self.dimension = dimension

        # Source locations
        self.source_locations = source_locations

        # Finite element type.
        allowed_ele_types = ['consistent', 'underintegrated']
        self.ele_type_eik = validate_parameter('ele_type_eik', ele_type_eik,
                                               allowed_ele_types)

        # Finite element order for the Eikonal analysis
        self.degree_eik = degree_eik if degree_eik is not None else \
            (2 if self.dimension == 2 else 1) if self.ele_type_eik == 'consistent' \
            else (4 if self.dimension == 2 else 3)

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

        # User solver tolerance
        self.tol = tol

    def eikonal_bcs(self, V: fire.FunctionSpace, lmin: float):
        """Impose Dirichlet BCs for eikonal equation.

        The source marker ``sou_marker`` is a product of narrow cosh^2 bumps, one
        per source, peak-normalized to 1. The dofs where ``sou_marker`` exceeds a
        high threshold are the source dofs. If more dofs than sources exceed the
        threshold (a source straddles two or more dofs), the global top-N dofs by
        marker value are selected, with ties broken deterministically by
        (lowest owning rank, lowest local index) — i.e. the "first coincidence".

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space where the boundary condition is applied.
        lmin : `float`
            Minimum mesh size.

        Returns
        -------
        bcs_eik : `list`
            Dirichlet BCs for eikonal.
        sou_marker : `Firedrake.Function`
            Function marking the source locations in the mesh.
        """
        comm = SpyroEnsemble.ensemble.comm
        rank = SpyroEnsemble.get_local_rank()

        # Penalty factor for the cosh (fdim * lmin is the minimum cell diameter)
        if self.dimension == 2:  # 2D
            fdim = 2**0.5
        if self.dimension == 3:  # 3D
            fdim = 3**0.5
        rf = np.round(np.arccosh(np.sqrt(1e4)) / (fdim * lmin), 0)

        sou_func = ""
        for source_coord in self.source_locations:

            zs, xs = source_coord[:2]
            expr = f"1 / (cosh({rf} * (z - {zs})) * cosh({rf}* (x - {xs})))**2"

            if self.dimension == 3:  # 3D
                ys = source_coord[2]
                expr += f"* 1 / (cosh({rf}* (y - {ys})))**2"

            sou_func += f"+ {expr}"

        sou_loc_func = generate_ufl_functions(self.mesh, sou_func, self.dimension)

        sou_marker = fire.Function(V, name="source_marker")
        sou_marker.assign(fire.assemble(fire.interpolate(sou_loc_func, V)))

        with sou_marker.dat.vec_ro as v:
            peak = v.max()[1]          # global MPI reduction

        if peak > 0:
            sou_marker.assign(sou_marker / peak)

        # Source DOFs
        threshold = 0.9999
        with sou_marker.dat.vec_ro as v:
            marker_vals = v.getArray().copy()  # owned dof values, order = V.dat.data_ro
        local_nodes = np.where(marker_vals > threshold)[0].astype(np.int32)

        # Per-rank sanity: the marker at matched dofs should be ~1
        if local_nodes.size > 0:
            vals_at_nodes = marker_vals[local_nodes]
            if np.any(vals_at_nodes < threshold):
                raise RuntimeError(f"[rank {rank}]: matched dofs below threshold.")

        # Global consistency: the number of matched dofs should equal the
        # number of sources (each source -> one dof above threshold).
        n_local = int(local_nodes.size)
        n_matched = comm.allreduce(n_local, op=MPI.SUM)
        n_src = len(self.source_locations)
        n_total = comm.allreduce(n_src, op=MPI.SUM)

        if n_matched < n_total:
            # Fewer dofs above threshold than sources: cannot proceed even by
            # snapping, because some source has no candidate dof at all.
            raise ValueError(
                f"Only {n_matched} dofs exceed threshold {threshold} for {n_total} "
                f"sources. r is likely too large (bump narrower than mesh spacing), "
                f"or a source is far from any dof. Cannot snap."
            )

        if n_matched > n_total:
            # More dofs than sources: select exactly n_total dofs globally,
            # ranked by marker value. Ties broken deterministically by
            # (lowest owning rank, lowest local index) — "first coincidence".
            local_vals = marker_vals[local_nodes]

            all_vals = comm.gather(local_vals, root=0)
            all_idx = comm.gather(local_nodes, root=0)
            all_rank = comm.gather(np.full(local_nodes.size, rank, dtype=np.int32), root=0)

            if rank == 0:
                cat_vals = np.concatenate(all_vals) if all_vals else np.array([])
                cat_idx = np.concatenate(all_idx) if all_idx else np.array([])
                cat_rank = np.concatenate(all_rank) if all_rank else np.array([])
                # lexsort: LAST key is primary.
                # Primary   : -cat_vals  (highest marker first)
                # Secondary :  cat_rank  (lowest owning rank first)
                # Tertiary  :  cat_idx   (lowest local index first)
                order = np.lexsort((cat_idx, cat_rank, -cat_vals))
                top = order[:n_total]

                top_idx = cat_idx[top].astype(np.int32)
                top_rank = cat_rank[top].astype(np.int32)

                # Distribute per-rank selections
                selected = {r: [] for r in range(comm.size)}
                for i, r in enumerate(top_rank):
                    selected[int(r)].append(int(top_idx[i]))
            else:
                selected = None

            # Broadcast the per-rank index lists (small: n_total entries total)
            selected = comm.bcast(selected, root=0)
            local_nodes = np.array(sorted(selected[comm.rank]), dtype=np.int32)

        # Final invariant: exactly n_total dofs selected across all ranks.
        n_selected = comm.allreduce(int(local_nodes.size), op=MPI.SUM)
        if n_selected != n_total:
            raise RuntimeError(f"Selected {n_selected} DOFs for {n_total} sources")

        # Overwrite sou_marker so that only the selected source dofs are 1
        # and all other dofs are 0. This produces a clean binary marker.
        with sou_marker.dat.vec as v:
            arr = v.getArray()          # writable view of owned dofs
            arr[:] = 0.0                # zero all owned dofs on this rank
            if local_nodes.size > 0:
                arr[local_nodes] = 1.0  # set selected source dofs to 1

        # Define BCs for eikonal
        bcs_eik = []
        if local_nodes.size > 0:
            bcs_eik.append(Dir_Point_BC(V, fire.Constant(0.0), local_nodes))

        else:
            raise ValueError("Error: Source Points Not Found!")

        return bcs_eik, sou_marker

    def define_int_dom(self, V):
        """Define the integration domain for the Eikonal equation.

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.

        Returns
        -------
        dx : `FiredrakeMeasure`
            Integration domain for the Eikonal equation.
        """

        if self.ele_type_eik == 'consistent':
            dx = fire.dx

        if self.ele_type_eik == 'underintegrated':
            quad_rule = quadrature_rules(V)[0]
            dx = fire.dx(**quad_rule)

        return dx

    def linear_eik(self, u, vy, c, V):
        """Assemble the linear Eikonal.

        Parameters
        ----------
        u : `Firedrake.TrialFunction`
            Trial function.
        vy : `Firedrake.TestFunction`
            Test function.
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.

        Returns
        -------
        FL : `Firedrake.Form`
            Linear Eikonal equation.
        """

        # Parameters
        f = fire.Constant(1.0)
        dx = self.define_int_dom(V)

        # Weak form
        lhs = fire.inner(fire.grad(u), fire.grad(vy)) * dx
        rhs = f / c * vy * dx
        FL = lhs - rhs

        return FL

    def nonlinear_eik(self, u, vy, c, V, diam_mesh, f_est=1.0):
        """Assemble the Nonlinear Eikonal with stabilizing term.

        Parameters
        ----------
        u : `Firedrake.TrialFunction`
            Trial function.
        vy : `Firedrake.TestFunction`
            Test function.
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal equation. Default is 1.0.

        Returns
        -------
        FNL: `Firedrake.Form`
            Nonlinear Eikonal equation.
        """

        # Parameters
        f = fire.Constant(1.0)
        dx = self.define_int_dom(V)

        # Stabilizer
        eps = fire.Constant(f_est) * diam_mesh

        # Weak form
        delta = fire.Constant(float_info.epsilon)  # float_info.min
        gr_norm = fire.sqrt(fire.inner(fire.grad(u), fire.grad(u))) + delta
        FNL = gr_norm * vy * dx - f / c * vy * dx + \
            eps * fire.inner(fire.grad(u), fire.grad(vy)) * dx

        return FNL

    @staticmethod
    def solver_opts(nl_solver='newtonls', l_solver='preonly',
                    user_atol=1e-16, user_iter=50, monitor=False):
        """Set the solver parameters.

        Parameters
        ----------
        nl_solver : `str`, optional
            Nonlinear solver type (See PETSC documentation).
        l_solver : `str`, optional
            Linear solver type.
        user_atol : `float`, optional
            Absolute user tolerance.
        user_iter : `float`, optional
            Maximum user iterations.
        monitor : 'bool', optional
            Prints the solver progress.

        Returns
        -------
        param_solver: `dict`
            Solver parameters

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/

        Tolerance Types
        ---------------
        atol: F(x) ≤ atol
        rtol: F(x) ≤ rtol∗F(x0)
        stol: || delta x || < stol*|| x ||
        haptol: lhs - rhs < haptol
        haptol < atol < rtol < stol
        """

        # Tolerances and iterations
        user_rtol = user_atol * 1e2
        user_stol = user_atol * 1e3
        ksp_max_it = user_iter

        param_solver = {'snes_type': nl_solver, 'ksp_type': l_solver}

        if nl_solver == 'newtontr':  # newton, cauchy, dogleg
            param_solver.update({'snes_tr_fallback_type': 'newton'})

        if nl_solver == 'ngmres':  # difference, none, linesearch
            param_solver.update({'snes_ngmres_select_type': 'linesearch'})

        if nl_solver == 'qn':
            param_solver.update({'snes_qn_m_type': 5,
                                 'snes_qn_powell_descent': True,
                                 # lbfgs, broyden, badbroyden
                                 'snes_qn_type': 'badbroyden',
                                 # diagonal, none, scalar, jacobian
                                 'snes_qn_scale_type': 'jacobian'})

        if nl_solver == 'ngs':
            param_solver.update({'snes_ngs_sweeps': 2,
                                 'snes_ngs_atol': user_atol,
                                 'snes_ngs_rtol': user_rtol,
                                 'snes_ngs_stol': user_stol,
                                 'snes_ngs_max_it': user_iter})

        if nl_solver == 'ncg':
            # fr, prp, dy, hs, cd
            param_solver.update({'snes_ncg_type': 'cd'})

        if l_solver == 'preonly':
            ig_nz = False
            pc_type = 'lu'  # lu, cholesky

            # mumps
            param_solver.update({'pc_factor_mat_solver_type': 'umfpack'})
        else:
            ig_nz = True
            pc_type = 'ilu'  # ilu, icc, lu, cholesky

        if l_solver == 'gmres':
            param_solver.update({'ksp_gmres_restart': 3,
                                 'ksp_gmres_haptol': user_atol * 1e-4})

        param_solver.update({
            'snes_linesearch_type': 'l2',  # l2, cp, basic
            'snes_linesearch_damping': 1.0,
            'snes_linesearch_maxlambda': 1.0,  # snes_linesearch_maxstep
            'snes_max_funcs': 1000,
            'snes_linesearch_order': 2,
            'snes_linesearch_alpha': 1e-4,
            'snes_max_it': user_iter,
            'snes_linesearch_rtol': user_rtol,
            'snes_linesearch_atol': user_atol,
            'snes_rtol': user_atol,
            'snes_atol': user_rtol,
            'snes_stol': user_stol,
            'ksp_max_it': ksp_max_it,
            'ksp_rtol': user_rtol,
            'ksp_atol': user_atol,
            'ksp_initial_guess_nonzero': ig_nz,
            'pc_type': pc_type,
            'pc_factor_reuse_ordering': True,
            'snes_monitor': None,
        })

        if monitor:  # For debugging
            param_solver.update({
                'snes_view': None,
                'snes_converged_reason': None,
                'snes_linesearch_monitor': None,
                'ksp_monitor_true_residual': None,
                'ksp_converged_reason': None,
                'report': True,
                'error_on_nonconvergence': True})

        return param_solver

    @staticmethod
    def initial_guess(c, c_min, V, diam_mesh, typ_igs='constant'):
        """Provide an initial guess for the Eikonal solver.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        c_min : `float`
            Minimum velocity value in the model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.
        typ_igs : `str`, optional
            Type of initial guess. 'constant' or 'variable'. Default is 'constant'.

        Returns
        -------
        init_guess : `float` or `ufl.algebra.Division`
            Initial guess for the Eikonal solver.
        """

        # Mesh cell diameters
        cell_diameter_function = fire.Function(V)
        cell_diameter_function.interpolate(diam_mesh)

        # Initial guess
        if typ_igs == 'constant':
            cell_diam_max = cell_diameter_function.dat.data_with_halos.max()
            init_guess = cell_diam_max / c_min

        elif typ_igs == 'variable':
            init_guess = cell_diameter_function / fire.Constant(c_min)

        else:
            validate_parameter('typ_igs', typ_igs, ['constant', 'variable'])

        return init_guess

    def linear_solution(self, wf_parameters, nl_solver='vinewtonssls',
                        l_solver='preonly', user_iter=50):
        """Solve the linear Eikonal equation.

        Parameters
        ----------
        wf_parameters : `list`
            List containing the weak form parameters.
            Structure: [u, vy, c, c_min, V, diam_mesh]
            - u : `Firedrake.TrialFunction`
                Trial function.
            - vy : `Firedrake.TestFunction`
                Test function.
            - c : `Firedrake.Function`
                Velocity model without absorbing layer.
            - c_min : `float`
                Minimum velocity value in the model without absorbing layer.
            - V : `Firedrake.FunctionSpace`
                Function space for the Eikonal modeling.
            - diam_mesh : `ufl.geometry.CellDiameter`
                Mesh cell diameters.
        nl_solver : `str`, optional
            Nonlinear solver type. Default is 'vinewtonssls'.
            Options: 'vinewtonssls', 'vinewtonrsls', 'newtonls', 'newtontr',
            'qn', 'ncg', 'ngs', 'ngmres' (See PETSC documentation).
        l_solver : `str`, optional.
            Linear solver type. Default is 'preonly'.
            Options: 'preonly', 'bcgs', 'gmres' (See PETSC documentation).
        user_iter : `int`, optional
            Maximum user iterations. Default is 50.

        Returns
        -------
        yp : `Firedrake.Function`
            Linear Eikonal field.

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/
        """

        # Weak form parameters
        u, vy, c, c_min, V, diam_mesh = wf_parameters

        # Initial guess
        yp = fire.Function(V, name='Eikonal (Time [s])')
        yp.assign(self.initial_guess(c, c_min, V, diam_mesh))

        # Linear Eikonal
        FeikL = self.linear_eik(u, vy, c, V)

        # Jacobian of the linear Eikonal
        J = fire.derivative(FeikL, yp)

        user_atol = self.tol**0.75
        while True:
            try:
                # Solver parameters
                p = self.solver_opts(nl_solver=nl_solver, l_solver=l_solver,
                                     user_atol=user_atol, user_iter=user_iter)

                # Solving LIN Eikonal
                fire.solve(fire.lhs(FeikL) == fire.rhs(FeikL), yp,
                           bcs=self.bcs_eik, solver_parameters=p, J=J)

                # Final parameters
                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + 'AbsTol: {:.1e}').format(
                    user_atol), flush=True)

                return yp

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)

                # Adjusting tolerance
                user_atol = user_atol * 10 if user_atol < 1e-5 \
                    else round(user_atol + 1e-5, 5)
                if user_atol > 1e-4:
                    print("Tolerance too high. Exiting.", flush=True)
                    break

    def nonlinear_solution(self, wf_parameters, nl_solver='vinewtonssls',
                           l_solver='preonly', user_iter=50, lin_sol=None):
        """Solve the nonlinear Eikonal equation.

        Parameters
        ----------
        wf_parameters : `list`
            List containing the weak form parameters.
            Structure: [vy, c, c_min, V, diam_mesh]
            - vy : `Firedrake.TestFunction`
                Test function.
            - c : `Firedrake.Function`
                Velocity model without absorbing layer.
            - c_min : `float`
                Minimum velocity value in the model without absorbing layer.
            - V : `Firedrake.FunctionSpace`
                Function space for the Eikonal modeling.
            - diam_mesh : `ufl.geometry.CellDiameter`
                Mesh cell diameters.
        nl_solver : `str`, optional
            Nonlinear solver type. Default is 'vinewtonssls'.
            Options: 'vinewtonssls', 'vinewtonrsls', 'newtonls', 'newtontr',
            'qn', 'ncg', 'ngs', 'ngmres' (See PETSC documentation).
        l_solver : `str`, optional.
            Linear solver type. Default is 'preonly'.
            Options: 'preonly', 'bcgs', 'gmres' (See PETSC documentation).
        user_iter : `int`, optional
            Maximum user iterations. Default is 50.
        lin_sol : `Firedrake.Function`, optional
            Linear Eikonal solution. Default is `None`.
            If `None`, an initial guess will be computed.

        Returns
        -------
        yp : `Firedrake.Function`
            Nonlinear Eikonal field.

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/
        """

        # Weak form parameters
        vy, c, c_min, V, diam_mesh = wf_parameters

        if lin_sol is None:
            # Initial guess for nonlinear Eikonal
            data_eikL = self.initial_guess(
                c, c_min, V, diam_mesh).dat.data_with_halos[:]
        else:
            # Clean numerical instabilities
            data_eikL = sanitize_num_array(lin_sol.dat.data_with_halos[:])

        user_atol = self.tol
        user_est = self.f_est
        while True:
            try:
                print(f"Iteration for Festab: {user_est:.2f}", flush=True)

                # Preserving intial guess
                yp = fire.Function(V, name='Eikonal (Time [s])')
                yp.dat.data_with_halos[:] = data_eikL

                # Solver parameters
                p = self.solver_opts(nl_solver=nl_solver, l_solver=l_solver,
                                     user_atol=user_atol, user_iter=user_iter)
                # Linear Eikonal
                FeikNL = self.nonlinear_eik(yp, vy, c, V, diam_mesh,
                                            f_est=user_est)

                # Jacobian of the nonlinear Eikonal
                J = fire.derivative(FeikNL, yp)

                # Solving NL Eikonal
                fire.solve(FeikNL == 0, yp, bcs=self.bcs_eik,
                           solver_parameters=p, J=J)

                # Final parameters
                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + 'AbsTol: {:.1e}').format(
                    user_atol), flush=True)
                print((solv_ok + 'Festab: {:.2f}').format(
                    user_est), flush=True)

                return yp

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)

                # Adjusting stabilizing factor
                user_est += 0.01
                if user_est > 1.0:
                    user_est = self.f_est
                    print("\nHigh Stabilizing Factor. Increasing Tolerance!",
                          flush=True)
                    user_atol = user_atol * 10 if user_atol < 1e-5 \
                        else round(user_atol + 1e-5, 5)
                    if user_atol > 1e-4:
                        print("High Tolerance. Exiting!", flush=True)
                        exit("No Results for Eikonal Equation")

    def eikonal_solver(self, c, c_min, V, diam_mesh):
        """Solve the Eikonal equation for model without absorbing layer.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        c_min : `float`
            Minimum velocity value in the model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.

        Returns
        -------
        yp : `Firedrake.Function`
            Eikonal field.
        """

        # Functions
        u = fire.TrialFunction(V)
        vy = fire.TestFunction(V)

        # Weak form parameters
        wf_parameters = [u, vy, c, c_min, V, diam_mesh]

        # Linear Eikonal
        print("\nSolving Pre-Eikonal", flush=True)
        yp = self.linear_solution(wf_parameters)

        # Nonlinear Eikonal
        print("\nSolving Post-Eikonal", flush=True)
        yp = self.nonlinear_solution(wf_parameters[1:], lin_sol=yp)

        return yp
